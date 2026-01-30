#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cointegration_tester.py - Core Co-integration Testing Module

Performs statistical co-integration tests for pair trading using:
- Augmented Dickey-Fuller (ADF) test
- Engle-Granger two-step method
- Half-life calculation for mean reversion

Designed for 5m timeframe to match V14s HTF bias strategy.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import warnings

# Suppress statsmodels warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.stattools import adfuller
except ImportError:
    print("ERROR: statsmodels not installed. Run: pip install statsmodels --break-system-packages")
    raise


@dataclass
class CointegrationResult:
    """Results from co-integration test"""
    is_cointegrated: bool
    p_value: float
    adf_statistic: float
    critical_values: Dict[str, float]
    half_life: float
    hedge_ratio: float
    confidence_level: float
    recommendation: str
    samples_used: int
    timeframe: str
    asset1: str
    asset2: str
    spread_std: float
    spread_mean: float


class CointegrationTester:
    """
    Co-integration testing for pair trading strategy
    
    Uses 5m timeframe to match HTF (Higher TimeFrame) bias in trading.
    Performs Engle-Granger two-step co-integration test.
    """
    
    def __init__(
        self, 
        significance_level: float = 0.05,
        min_samples: int = 200,
        timeframe: str = "5m"
    ):
        """
        Initialize co-integration tester
        
        Args:
            significance_level: P-value threshold (default 0.05 = 5%)
            min_samples: Minimum data points required (default 200)
            timeframe: Trading timeframe (default "5m")
        """
        self.significance_level = significance_level
        self.min_samples = min_samples
        self.timeframe = timeframe
        
        # Timeframe conversion to minutes
        self.timeframe_minutes = self._parse_timeframe(timeframe)
    
    def _parse_timeframe(self, tf: str) -> int:
        """Convert timeframe string to minutes"""
        if tf == "5s":
            return 5 / 60
        elif tf == "1m":
            return 1
        elif tf == "5m":
            return 5
        elif tf == "15m":
            return 15
        elif tf == "1h":
            return 60
        else:
            return 5  # Default to 5m
    
    def test_cointegration(
        self, 
        asset1: str,
        asset2: str,
        asset1_prices: np.ndarray,
        asset2_prices: np.ndarray
    ) -> CointegrationResult:
        """
        Perform Engle-Granger co-integration test
        
        Args:
            asset1: Name of first asset (e.g., "SOL")
            asset2: Name of second asset (e.g., "MELANIA")
            asset1_prices: Price series for asset1
            asset2_prices: Price series for asset2
        
        Returns:
            CointegrationResult with test statistics and recommendation
        """
        
        # Validation
        if len(asset1_prices) != len(asset2_prices):
            raise ValueError(f"Price series length mismatch: {len(asset1_prices)} vs {len(asset2_prices)}")
        
        if len(asset1_prices) < self.min_samples:
            return self._create_insufficient_data_result(asset1, asset2, len(asset1_prices))
        
        # Step 1: Calculate hedge ratio (beta) using OLS regression
        hedge_ratio = self._calculate_hedge_ratio(asset1_prices, asset2_prices)
        
        # Step 2: Calculate spread (residuals)
        spread = asset2_prices - hedge_ratio * asset1_prices
        
        # Step 3: Test spread stationarity using ADF test
        try:
            adf_result = adfuller(spread, maxlag=1, regression='c')
            adf_statistic = adf_result[0]
            p_value = adf_result[1]
            critical_values = adf_result[4]
        except Exception as e:
            print(f"ERROR in ADF test: {e}")
            return self._create_error_result(asset1, asset2, str(e))
        
        # Step 4: Calculate half-life of mean reversion
        half_life = self._calculate_half_life(spread)
        
        # Step 5: Determine if cointegrated
        is_cointegrated = p_value < self.significance_level
        
        # Step 6: Get recommendation
        recommendation = self._get_recommendation(is_cointegrated, p_value, half_life)
        
        # Step 7: Calculate spread statistics
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)
        
        return CointegrationResult(
            is_cointegrated=is_cointegrated,
            p_value=p_value,
            adf_statistic=adf_statistic,
            critical_values=critical_values,
            half_life=half_life,
            hedge_ratio=hedge_ratio,
            confidence_level=1 - self.significance_level,
            recommendation=recommendation,
            samples_used=len(asset1_prices),
            timeframe=self.timeframe,
            asset1=asset1,
            asset2=asset2,
            spread_std=spread_std,
            spread_mean=spread_mean
        )
    
    def _calculate_hedge_ratio(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate optimal hedge ratio (beta) via OLS regression
        
        Regression: y = alpha + beta * x + epsilon
        Returns: beta (hedge ratio)
        """
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0.0
        
        beta = numerator / denominator
        return beta
    
    def _calculate_half_life(self, spread: np.ndarray) -> float:
        """
        Calculate mean-reversion half-life using AR(1) model
        
        AR(1): Δspread_t = λ * spread_{t-1} + ε
        Half-life = -ln(2) / λ
        
        Returns:
            Half-life in number of bars (e.g., 5m bars)
            Returns np.inf if no mean reversion detected
        """
        spread_lag = spread[:-1]
        spread_diff = np.diff(spread)
        
        # Remove any NaN or inf values
        valid_indices = np.isfinite(spread_lag) & np.isfinite(spread_diff)
        spread_lag = spread_lag[valid_indices]
        spread_diff = spread_diff[valid_indices]
        
        if len(spread_lag) < 10:
            return np.inf
        
        # AR(1) regression: Δspread = λ * spread_lag + constant
        try:
            coeffs = np.polyfit(spread_lag, spread_diff, 1)
            lambda_param = coeffs[0]
        except:
            return np.inf
        
        # If lambda >= 0, no mean reversion
        if lambda_param >= 0:
            return np.inf
        
        # Calculate half-life
        half_life = -np.log(2) / lambda_param
        
        # Sanity check: if half-life is negative or too large, return inf
        if half_life < 0 or half_life > 1000:
            return np.inf
        
        return half_life
    
    def _get_recommendation(
        self, 
        is_cointegrated: bool, 
        p_value: float, 
        half_life: float
    ) -> str:
        """
        Generate trading recommendation based on test results
        
        Categories:
        - EXCELLENT: Strong cointegration + fast mean reversion
        - GOOD: Cointegrated + moderate mean reversion
        - MODERATE: Cointegrated but slow mean reversion
        - WEAK: Marginal cointegration
        - NOT_SUITABLE: Not cointegrated
        """
        if not is_cointegrated:
            return "NOT_SUITABLE"
        
        # Based on 5m bars:
        # 6 bars = 30 minutes
        # 12 bars = 1 hour
        # 24 bars = 2 hours
        # 72 bars = 6 hours
        # 288 bars = 1 day
        
        if p_value < 0.01:  # Very strong cointegration (1% level)
            if half_life < 6:
                return "EXCELLENT"
            elif half_life < 24:
                return "GOOD"
            elif half_life < 72:
                return "MODERATE"
            else:
                return "WEAK"
        
        elif p_value < 0.05:  # Strong cointegration (5% level)
            if half_life < 12:
                return "GOOD"
            elif half_life < 48:
                return "MODERATE"
            else:
                return "WEAK"
        
        else:  # Marginal cointegration
            return "WEAK"
    
    def _create_insufficient_data_result(
        self, 
        asset1: str, 
        asset2: str, 
        samples: int
    ) -> CointegrationResult:
        """Create result object for insufficient data"""
        return CointegrationResult(
            is_cointegrated=False,
            p_value=1.0,
            adf_statistic=0.0,
            critical_values={},
            half_life=np.inf,
            hedge_ratio=0.0,
            confidence_level=1 - self.significance_level,
            recommendation="INSUFFICIENT_DATA",
            samples_used=samples,
            timeframe=self.timeframe,
            asset1=asset1,
            asset2=asset2,
            spread_std=0.0,
            spread_mean=0.0
        )
    
    def _create_error_result(
        self, 
        asset1: str, 
        asset2: str, 
        error_msg: str
    ) -> CointegrationResult:
        """Create result object for test errors"""
        return CointegrationResult(
            is_cointegrated=False,
            p_value=1.0,
            adf_statistic=0.0,
            critical_values={},
            half_life=np.inf,
            hedge_ratio=0.0,
            confidence_level=1 - self.significance_level,
            recommendation="ERROR",
            samples_used=0,
            timeframe=self.timeframe,
            asset1=asset1,
            asset2=asset2,
            spread_std=0.0,
            spread_mean=0.0
        )


# Helper function for quick testing
def quick_test(asset1_prices: np.ndarray, asset2_prices: np.ndarray) -> Dict:
    """Quick co-integration test with default parameters"""
    tester = CointegrationTester()
    result = tester.test_cointegration("ASSET1", "ASSET2", asset1_prices, asset2_prices)
    
    return {
        'cointegrated': result.is_cointegrated,
        'p_value': result.p_value,
        'half_life': result.half_life,
        'hedge_ratio': result.hedge_ratio,
        'recommendation': result.recommendation
    }


if __name__ == "__main__":
    # Simple test with synthetic data
    print("Testing CointegrationTester with synthetic data...")
    
    # Generate cointegrated pair
    np.random.seed(42)
    n = 500
    asset1 = np.cumsum(np.random.randn(n)) + 100
    asset2 = 2.5 * asset1 + np.random.randn(n) * 0.5  # Cointegrated with asset1
    
    tester = CointegrationTester()
    result = tester.test_cointegration("TEST1", "TEST2", asset1, asset2)
    
    print(f"\nResults:")
    print(f"  Cointegrated: {result.is_cointegrated}")
    print(f"  P-value: {result.p_value:.4f}")
    print(f"  Half-life: {result.half_life:.1f} bars")
    print(f"  Hedge Ratio: {result.hedge_ratio:.4f}")
    print(f"  Recommendation: {result.recommendation}")
