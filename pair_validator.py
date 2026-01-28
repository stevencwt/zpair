#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pair_validator.py - Pair Validation Orchestrator (STRICT MODE)

Loads price data and orchestrates co-integration testing for multiple pairs.
Integrates with existing V14s data infrastructure (Telemetry Layer).

STRICT MODE: Synthetic/Mock data generation has been disabled.
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from cointegration_tester import CointegrationTester, CointegrationResult

# [NEW] Integration with V14s Architecture
try:
    from config_manager import load_config
    from telemetry_provider import get_telemetry_provider
    V14S_MODULES_AVAILABLE = True
except ImportError:
    V14S_MODULES_AVAILABLE = False
    print("⚠️  WARNING: V14s modules (config_manager, telemetry_provider) not found.")


class PairValidator:
    """
    Validates trading pairs using co-integration testing
    
    Data Source Priority:
    1. Local Cached Files (fastest)
    2. Hyperliquid API via TelemetryProvider (real data)
    """
    
    def __init__(
        self, 
        timeframe: str = "5m",
        lookback_bars: int = 1000,
        significance_level: float = 0.05,
        data_directory: str = ".",
        config_path: str = "config.json"
    ):
        self.timeframe = timeframe
        self.lookback_bars = lookback_bars
        self.data_directory = Path(data_directory)
        self.config_path = config_path
        
        # Initialize co-integration tester
        self.coint_tester = CointegrationTester(
            significance_level=significance_level,
            timeframe=timeframe
        )
        
        # Lazy loader for the provider
        self.provider = None

    def _get_provider(self):
        """Lazy initialization of TelemetryProvider to avoid overhead if not needed"""
        if self.provider:
            return self.provider
            
        if not V14S_MODULES_AVAILABLE:
            return None

        try:
            print("   Initializing Telemetry Provider...")
            # Load config using your manager
            config = load_config(self.config_path)
            # Use your factory to get the provider
            self.provider = get_telemetry_provider(config)
            return self.provider
        except Exception as e:
            print(f"   ⚠️  Failed to initialize provider: {e}")
            return None
    
    def validate_pair(
        self, 
        asset1: str, 
        asset2: str,
        asset1_prices: Optional[np.ndarray] = None,
        asset2_prices: Optional[np.ndarray] = None
    ) -> CointegrationResult:
        """
        Validate a single pair
        """
        
        # Load price data if not provided
        if asset1_prices is None or asset2_prices is None:
            asset1_prices, asset2_prices = self._load_pair_data(asset1, asset2)
        
        # Validate data
        if asset1_prices is None or asset2_prices is None:
            print(f"ERROR: Could not load data for {asset1}/{asset2}")
            return self.coint_tester._create_error_result(asset1, asset2, "Data loading failed")
        
        if len(asset1_prices) == 0 or len(asset2_prices) == 0:
            print(f"ERROR: Empty price data for {asset1}/{asset2}")
            return self.coint_tester._create_error_result(asset1, asset2, "Empty data")
        
        # Run co-integration test
        result = self.coint_tester.test_cointegration(
            asset1, asset2, asset1_prices, asset2_prices
        )
        
        return result
    
    def validate_multiple_pairs(
        self, 
        pairs: List[Dict[str, str]]
    ) -> List[CointegrationResult]:
        """Validate multiple pairs from config"""
        results = []
        
        for i, pair in enumerate(pairs, 1):
            asset1 = pair.get('asset1')
            asset2 = pair.get('asset2')
            
            if not asset1 or not asset2:
                print(f"WARNING: Skipping invalid pair config: {pair}")
                continue
            
            print(f"\n[{i}/{len(pairs)}] Testing {asset1}/{asset2}...")
            
            try:
                result = self.validate_pair(asset1, asset2)
                results.append(result)
            except Exception as e:
                print(f"ERROR testing {asset1}/{asset2}: {e}")
                error_result = self.coint_tester._create_error_result(
                    asset1, asset2, str(e)
                )
                results.append(error_result)
        
        return results
    
    def _load_pair_data(
        self, 
        asset1: str, 
        asset2: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load price data with STRICT strategy (No Synthetic Fallback)
        """
        
        # 1. Try Aggregated Data (Highest Priority)
        prices = self._load_from_aggregated_data(asset1, asset2)
        if prices is not None:
            return prices
        
        # 2. Try Local Cache Files
        prices = self._load_from_separate_files(asset1, asset2)
        if prices is not None:
            return prices

        # 3. Try Fetching from TelemetryProvider (Real API)
        prices = self._fetch_from_provider(asset1, asset2)
        if prices is not None:
            return prices
        
        # 4. STRICT FAILURE (No Mock Data)
        print(f"❌ CRITICAL ERROR: No real data found for {asset1}/{asset2}.")
        print(f"   Please check your API connection or config.json.")
        return None, None

    def _fetch_from_provider(self, asset1: str, asset2: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Fetch fresh data from Hyperliquid via TelemetryProvider"""
        provider = self._get_provider()
        if not provider:
            return None

        print(f"   Fetching real data from Hyperliquid ({self.lookback_bars} bars)...")
        
        # Fetch individual assets
        # We request slightly more to ensure we have enough after alignment
        fetch_count = int(self.lookback_bars * 1.2) 
        
        # Fetch Asset 1
        data1 = provider.get_candles(asset1, self.timeframe, fetch_count)
        if not data1:
            print(f"   ❌ Failed to fetch {asset1}")
            return None
        
        # Fetch Asset 2
        time.sleep(0.2) # Rate limit politeness
        data2 = provider.get_candles(asset2, self.timeframe, fetch_count)
        if not data2:
            print(f"   ❌ Failed to fetch {asset2}")
            return None

        # Extract closing prices and timestamps for alignment
        # We need to ensure we align them by time
        df1_prices = {d['t']: float(d['c']) for d in data1}
        df2_prices = {d['t']: float(d['c']) for d in data2}
        
        # Find common timestamps
        common_times = sorted(list(set(df1_prices.keys()) & set(df2_prices.keys())))
        
        if len(common_times) < self.lookback_bars * 0.5:
             print(f"   ❌ Insufficient overlapping data: {len(common_times)} common bars")
             return None

        # Create aligned arrays
        prices1 = np.array([df1_prices[t] for t in common_times])
        prices2 = np.array([df2_prices[t] for t in common_times])
        
        # Cache the data to disk so next run is fast
        self._save_to_cache(asset1, prices1)
        self._save_to_cache(asset2, prices2)
        
        # Trim to requested lookback
        if len(prices1) > self.lookback_bars:
            prices1 = prices1[-self.lookback_bars:]
            prices2 = prices2[-self.lookback_bars:]
            
        print(f"   ✅ Fetched and cached {len(prices1)} aligned bars.")
        return prices1, prices2

    def _save_to_cache(self, asset: str, prices: np.ndarray):
        """Save fetched data to JSON for faster local loading next time"""
        filename = self.data_directory / f"{asset}_{self.timeframe}.json"
        data = {
            "symbol": asset,
            "timeframe": self.timeframe,
            "prices": prices.tolist(),
            "generated_at": datetime.now().isoformat()
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

    def _load_from_aggregated_data(self, asset1: str, asset2: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load from aggregated data directory"""
        possible_dirs = [
            self.data_directory / f"aggregated_{asset1}_{asset2}_5m1m",
            self.data_directory / f"aggregated_{asset1}_{asset2}",
            self.data_directory / f"marketstate_{asset1}_{asset2}_5m1m",
        ]
        
        for data_dir in possible_dirs:
            if data_dir.exists():
                json_file = data_dir / f"marketstate_{self.timeframe}.json"
                if json_file.exists():
                    try:
                        return self._parse_aggregated_json(json_file, asset1, asset2)
                    except Exception as e:
                        print(f"WARNING: Failed to parse {json_file}: {e}")
        return None
    
    def _parse_aggregated_json(self, json_file: Path, asset1: str, asset2: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Parse aggregated JSON file"""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if 'prices' in data and 'history' in data['prices']:
                history = data['prices']['history']
                asset1_prices = np.array([p.get('primary', 0) for p in history])
                asset2_prices = np.array([p.get('reference', 0) for p in history])
                
                if len(asset1_prices) > self.lookback_bars:
                    asset1_prices = asset1_prices[-self.lookback_bars:]
                    asset2_prices = asset2_prices[-self.lookback_bars:]
                
                return asset1_prices, asset2_prices
        except Exception as e:
            print(f"ERROR parsing {json_file}: {e}")
        return None
    
    def _load_from_separate_files(self, asset1: str, asset2: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load from separate asset files"""
        asset1_file = self.data_directory / f"{asset1}_{self.timeframe}.json"
        asset2_file = self.data_directory / f"{asset2}_{self.timeframe}.json"
        
        if asset1_file.exists() and asset2_file.exists():
            try:
                with open(asset1_file, 'r') as f:
                    asset1_data = json.load(f)
                with open(asset2_file, 'r') as f:
                    asset2_data = json.load(f)
                
                asset1_prices = np.array(asset1_data.get('prices', []))
                asset2_prices = np.array(asset2_data.get('prices', []))
                
                if len(asset1_prices) > self.lookback_bars:
                    asset1_prices = asset1_prices[-self.lookback_bars:]
                    asset2_prices = asset2_prices[-self.lookback_bars:]
                
                return asset1_prices, asset2_prices
            except Exception as e:
                print(f"ERROR loading separate files: {e}")
        return None
    
    def export_results_to_json(
        self, 
        results: List[CointegrationResult], 
        filename: str = "cointegration_results.json"
    ):
        """Export test results to JSON file"""
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'timeframe': self.timeframe,
            'lookback_bars': self.lookback_bars,
            'pairs_tested': len(results),
            'results': []
        }
        
        for result in results:
            output_data['results'].append({
                'asset1': result.asset1,
                'asset2': result.asset2,
                'pair': f"{result.asset1}/{result.asset2}",
                'is_cointegrated': bool(result.is_cointegrated),
                'p_value': float(result.p_value),
                'adf_statistic': float(result.adf_statistic),
                'half_life_bars': float(result.half_life) if not np.isinf(result.half_life) else None,
                'half_life_minutes': float(result.half_life * 5) if not np.isinf(result.half_life) else None,
                'hedge_ratio': float(result.hedge_ratio),
                'recommendation': result.recommendation,
                'samples_used': result.samples_used
            })
        
        output_path = self.data_directory / filename
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✅ Results exported to: {output_path}")

# Run the test
if __name__ == "__main__":
    print("Testing PairValidator (STRICT MODE)...")
    validator = PairValidator(timeframe="5m", lookback_bars=200)
    result = validator.validate_pair("SOL", "MELANIA")
    
    if result:
        print(f"\nResults:")
        print(f"  Pair: {result.asset1}/{result.asset2}")
        print(f"  Cointegrated: {result.is_cointegrated}")
        print(f"  P-value: {result.p_value:.4f}")
    else:
        print("\n❌ Validation Failed (No Data)")