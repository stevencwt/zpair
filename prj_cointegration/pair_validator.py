#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pair_validator.py - Pair Validation Orchestrator

Loads price data and orchestrates co-integration testing for multiple pairs.
Integrates with existing V14s data infrastructure.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

from cointegration_tester import CointegrationTester, CointegrationResult


class PairValidator:
    """
    Validates trading pairs using co-integration testing
    
    Loads data from existing V14s infrastructure:
    - Multi-timeframe buffers
    - Aggregated data files
    - Telemetry provider
    """
    
    def __init__(
        self, 
        timeframe: str = "5m",
        lookback_bars: int = 1000,
        significance_level: float = 0.05,
        data_directory: str = "."
    ):
        """
        Initialize pair validator
        
        Args:
            timeframe: Timeframe for testing (default "5m")
            lookback_bars: Number of bars to use (default 1000)
            significance_level: P-value threshold (default 0.05)
            data_directory: Base directory for data files
        """
        self.timeframe = timeframe
        self.lookback_bars = lookback_bars
        self.data_directory = Path(data_directory)
        
        # Initialize co-integration tester
        self.coint_tester = CointegrationTester(
            significance_level=significance_level,
            timeframe=timeframe
        )
    
    def validate_pair(
        self, 
        asset1: str, 
        asset2: str,
        asset1_prices: Optional[np.ndarray] = None,
        asset2_prices: Optional[np.ndarray] = None
    ) -> CointegrationResult:
        """
        Validate a single pair
        
        Args:
            asset1: First asset symbol (e.g., "SOL")
            asset2: Second asset symbol (e.g., "MELANIA")
            asset1_prices: Optional pre-loaded price data for asset1
            asset2_prices: Optional pre-loaded price data for asset2
        
        Returns:
            CointegrationResult with test statistics
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
        """
        Validate multiple pairs from config
        
        Args:
            pairs: List of pair dicts, e.g., [{"asset1": "SOL", "asset2": "MELANIA"}]
        
        Returns:
            List of CointegrationResult objects
        """
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
        Load price data for a pair
        
        Tries multiple data sources in order:
        1. Aggregated data files (e.g., aggregated_SOL_MELANIA_5m1m/)
        2. Multi-timeframe buffer files
        3. Synthetic/demo data for testing
        
        Returns:
            Tuple of (asset1_prices, asset2_prices) or (None, None) if failed
        """
        
        # Try loading from aggregated data directory
        prices = self._load_from_aggregated_data(asset1, asset2)
        if prices is not None:
            return prices
        
        # Try loading from separate asset files
        prices = self._load_from_separate_files(asset1, asset2)
        if prices is not None:
            return prices
        
        # Generate synthetic data for testing (fallback)
        print(f"WARNING: Using synthetic data for {asset1}/{asset2} (no real data found)")
        return self._generate_synthetic_data()
    
    def _load_from_aggregated_data(
        self, 
        asset1: str, 
        asset2: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Load from aggregated data directory
        Example: aggregated_SOL_MELANIA_5m1m/marketstate_5m.json
        """
        
        # Try different directory patterns
        possible_dirs = [
            self.data_directory / f"aggregated_{asset1}_{asset2}_5m1m",
            self.data_directory / f"aggregated_{asset1}_{asset2}",
            self.data_directory / f"marketstate_{asset1}_{asset2}_5m1m",
        ]
        
        for data_dir in possible_dirs:
            if data_dir.exists():
                # Try to load market state or buffer data
                json_file = data_dir / f"marketstate_{self.timeframe}.json"
                if json_file.exists():
                    try:
                        return self._parse_aggregated_json(json_file, asset1, asset2)
                    except Exception as e:
                        print(f"WARNING: Failed to parse {json_file}: {e}")
        
        return None
    
    def _parse_aggregated_json(
        self, 
        json_file: Path, 
        asset1: str, 
        asset2: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Parse aggregated JSON file for price data"""
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract price arrays from JSON
            # Structure depends on your actual data format
            # Adapt this based on your actual JSON structure
            
            if 'prices' in data and 'history' in data['prices']:
                history = data['prices']['history']
                asset1_prices = np.array([p.get('primary', 0) for p in history])
                asset2_prices = np.array([p.get('reference', 0) for p in history])
                
                # Limit to lookback_bars
                if len(asset1_prices) > self.lookback_bars:
                    asset1_prices = asset1_prices[-self.lookback_bars:]
                    asset2_prices = asset2_prices[-self.lookback_bars:]
                
                return asset1_prices, asset2_prices
            
        except Exception as e:
            print(f"ERROR parsing {json_file}: {e}")
        
        return None
    
    def _load_from_separate_files(
        self, 
        asset1: str, 
        asset2: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Load from separate asset files
        Example: data/SOL_5m.json, data/MELANIA_5m.json
        """
        
        asset1_file = self.data_directory / f"{asset1}_{self.timeframe}.json"
        asset2_file = self.data_directory / f"{asset2}_{self.timeframe}.json"
        
        if asset1_file.exists() and asset2_file.exists():
            try:
                with open(asset1_file, 'r') as f:
                    asset1_data = json.load(f)
                with open(asset2_file, 'r') as f:
                    asset2_data = json.load(f)
                
                # Extract prices (adapt to your data format)
                asset1_prices = np.array(asset1_data.get('prices', []))
                asset2_prices = np.array(asset2_data.get('prices', []))
                
                # Limit to lookback_bars
                if len(asset1_prices) > self.lookback_bars:
                    asset1_prices = asset1_prices[-self.lookback_bars:]
                    asset2_prices = asset2_prices[-self.lookback_bars:]
                
                return asset1_prices, asset2_prices
                
            except Exception as e:
                print(f"ERROR loading separate files: {e}")
        
        return None
    
    def _generate_synthetic_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic cointegrated data for testing
        This is a fallback when real data is not available
        """
        np.random.seed(42)
        n = self.lookback_bars
        
        # Generate random walk for asset1
        asset1 = np.cumsum(np.random.randn(n) * 0.5) + 100
        
        # Generate cointegrated asset2
        hedge_ratio = 150 + np.random.randn() * 20  # Random hedge ratio
        asset2 = hedge_ratio * asset1 + np.cumsum(np.random.randn(n) * 2)
        
        return asset1, asset2
    
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
                'is_cointegrated': result.is_cointegrated,
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


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing PairValidator...")
    
    validator = PairValidator(timeframe="5m", lookback_bars=500)
    
    # Test single pair
    result = validator.validate_pair("SOL", "MELANIA")
    
    print(f"\nResults:")
    print(f"  Pair: {result.asset1}/{result.asset2}")
    print(f"  Cointegrated: {result.is_cointegrated}")
    print(f"  P-value: {result.p_value:.4f}")
    print(f"  Half-life: {result.half_life:.1f} bars")
    print(f"  Recommendation: {result.recommendation}")
