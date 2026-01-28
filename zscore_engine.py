#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module B - Z-Score Engine
Evaluates how unusual the current spread is compared to its recent history.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import warnings

class ZScoreEngine:
    """
    Z-Score calculation engine for pair trading analysis.
    
    This module evaluates how unusual the current spread is compared to its own recent history.
    It calculates the average and variation of the spread over a chosen lookback period
    and expresses the current deviation as a Z-score (in standard deviation units).
    """
    
    def __init__(self, window_size: int = 60, min_periods: int = 30, 
                 calculation_method: str = 'rolling', center: bool = False, ddof: int = 1):
        """
        Initialize the Z-Score engine.
        
        Parameters:
        -----------
        window_size : int
            Rolling window size for z-score calculation
        min_periods : int
            Minimum number of periods required for calculation
        calculation_method : str
            Method for calculation ('rolling' or 'expanding')
        center : bool
            Whether to center the rolling window
        ddof : int
            Delta degrees of freedom for standard deviation calculation
        """
        self.window_size = window_size
        self.min_periods = min_periods
        self.calculation_method = calculation_method
        self.center = center
        self.ddof = ddof
        
    def calculate_zscore(self, spread_series: pd.Series) -> pd.Series:
        """
        Calculate z-score for a spread series.
        
        Parameters:
        -----------
        spread_series : pd.Series
            The spread or residuals series
            
        Returns:
        --------
        pd.Series containing z-scores
        """
        if len(spread_series) < self.min_periods:
            warnings.warn(f"Insufficient data for z-score calculation: {len(spread_series)} < {self.min_periods}")
            return pd.Series(index=spread_series.index, dtype=float)
        
        if self.calculation_method == 'rolling':
            return self._calculate_rolling_zscore(spread_series)
        elif self.calculation_method == 'expanding':
            return self._calculate_expanding_zscore(spread_series)
        else:
            raise ValueError(f"Unknown calculation method: {self.calculation_method}")
    
    def _calculate_rolling_zscore(self, spread_series: pd.Series) -> pd.Series:
        """Calculate rolling z-score."""
        rolling_mean = spread_series.rolling(
            window=self.window_size,
            min_periods=self.min_periods,
            center=self.center
        ).mean()
        
        rolling_std = spread_series.rolling(
            window=self.window_size,
            min_periods=self.min_periods,
            center=self.center
        ).std(ddof=self.ddof)
        
        # Calculate z-score, handling division by zero
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            zscore = (spread_series - rolling_mean) / rolling_std
        
        # Replace inf and -inf with NaN
        zscore = zscore.replace([np.inf, -np.inf], np.nan)
        
        return zscore
    
    def _calculate_expanding_zscore(self, spread_series: pd.Series) -> pd.Series:
        """Calculate expanding z-score."""
        expanding_mean = spread_series.expanding(
            min_periods=self.min_periods
        ).mean()
        
        expanding_std = spread_series.expanding(
            min_periods=self.min_periods
        ).std(ddof=self.ddof)
        
        # Calculate z-score, handling division by zero
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            zscore = (spread_series - expanding_mean) / expanding_std
        
        # Replace inf and -inf with NaN
        zscore = zscore.replace([np.inf, -np.inf], np.nan)
        
        return zscore
    
    def calculate_zscore_from_ratio(self, ratio_series: pd.Series) -> pd.Series:
        """
        Calculate z-score directly from a ratio series.
        
        This method is compatible with the existing dataplotter implementation.
        
        Parameters:
        -----------
        ratio_series : pd.Series
            Price ratio series (asset1/asset2)
            
        Returns:
        --------
        pd.Series containing z-scores
        """
        return self.calculate_zscore(ratio_series)
    
    def get_zscore_levels(self, zscore_series: pd.Series, thresholds: List[float]) -> Dict:
        """
        Analyze z-score levels and threshold breaches.
        
        Parameters:
        -----------
        zscore_series : pd.Series
            Z-score series
        thresholds : List[float]
            List of threshold levels to analyze
            
        Returns:
        --------
        Dict containing level analysis
        """
        clean_zscore = zscore_series.dropna()
        
        if len(clean_zscore) == 0:
            return {'error': 'No valid z-score data'}
        
        current_zscore = clean_zscore.iloc[-1]
        
        results = {
            'current_zscore': current_zscore,
            'zscore_stats': {
                'mean': clean_zscore.mean(),
                'std': clean_zscore.std(),
                'min': clean_zscore.min(),
                'max': clean_zscore.max(),
                'median': clean_zscore.median()
            },
            'threshold_analysis': {},
            'current_level': self._classify_zscore_level(current_zscore, thresholds)
        }
        
        # Analyze each threshold
        for threshold in thresholds:
            positive_breaches = (clean_zscore > threshold).sum()
            negative_breaches = (clean_zscore < -threshold).sum()
            total_observations = len(clean_zscore)
            
            results['threshold_analysis'][f'{threshold}sigma'] = {
                'positive_breaches': positive_breaches,
                'negative_breaches': negative_breaches,
                'total_breaches': positive_breaches + negative_breaches,
                'positive_breach_pct': (positive_breaches / total_observations) * 100,
                'negative_breach_pct': (negative_breaches / total_observations) * 100,
                'total_breach_pct': ((positive_breaches + negative_breaches) / total_observations) * 100
            }
        
        return results
    
    def _classify_zscore_level(self, zscore_value: float, thresholds: List[float]) -> str:
        """Classify the current z-score level based on thresholds."""
        if np.isnan(zscore_value):
            return 'undefined'
        
        abs_zscore = abs(zscore_value)
        direction = 'positive' if zscore_value > 0 else 'negative'
        
        # Sort thresholds in ascending order
        sorted_thresholds = sorted(thresholds)
        
        for i, threshold in enumerate(sorted_thresholds):
            if abs_zscore <= threshold:
                if i == 0:
                    return f'{direction}_low' if abs_zscore > 0.5 else 'neutral'
                else:
                    return f'{direction}_level_{i}'
        
        # If beyond all thresholds
        return f'{direction}_extreme'
    
    def calculate_zscore_velocity(self, zscore_series: pd.Series, periods: int = 5) -> pd.Series:
        """
        Calculate the rate of change (velocity) of z-score.
        
        Parameters:
        -----------
        zscore_series : pd.Series
            Z-score series
        periods : int
            Number of periods for velocity calculation
            
        Returns:
        --------
        pd.Series containing z-score velocity
        """
        return zscore_series.diff(periods) / periods
    
    def calculate_zscore_momentum(self, zscore_series: pd.Series, short_window: int = 5, 
                                long_window: int = 20) -> pd.Series:
        """
        Calculate z-score momentum using moving average crossover.
        
        Parameters:
        -----------
        zscore_series : pd.Series
            Z-score series
        short_window : int
            Short-term moving average window
        long_window : int
            Long-term moving average window
            
        Returns:
        --------
        pd.Series containing momentum indicator
        """
        short_ma = zscore_series.rolling(window=short_window, min_periods=1).mean()
        long_ma = zscore_series.rolling(window=long_window, min_periods=1).mean()
        
        momentum = short_ma - long_ma
        return momentum
    
    def get_mean_reversion_signals(self, zscore_series: pd.Series, 
                                 entry_threshold: float = 2.0, 
                                 exit_threshold: float = 0.5) -> Dict:
        """
        Generate mean reversion trading signals based on z-score levels.
        
        Parameters:
        -----------
        zscore_series : pd.Series
            Z-score series
        entry_threshold : float
            Z-score level to trigger entry signals
        exit_threshold : float
            Z-score level to trigger exit signals
            
        Returns:
        --------
        Dict containing signal information
        """
        clean_zscore = zscore_series.dropna()
        
        if len(clean_zscore) == 0:
            return {'error': 'No valid z-score data for signal generation'}
        
        current_zscore = clean_zscore.iloc[-1]
        
        # Determine signal type
        signal = 'hold'
        signal_strength = abs(current_zscore)
        
        if current_zscore > entry_threshold:
            signal = 'short_entry'  # Asset A overvalued relative to Asset B
        elif current_zscore < -entry_threshold:
            signal = 'long_entry'   # Asset A undervalued relative to Asset B
        elif abs(current_zscore) < exit_threshold:
            signal = 'exit'         # Near fair value, consider closing positions
        
        return {
            'current_zscore': current_zscore,
            'signal': signal,
            'signal_strength': signal_strength,
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
            'confidence': min(signal_strength / entry_threshold, 1.0) if signal_strength > exit_threshold else 0.0
        }
    
    def calculate_percentile_ranks(self, zscore_series: pd.Series, 
                                 lookback_periods: Optional[int] = None) -> Dict:
        """
        Calculate percentile ranks for current z-score relative to history.
        
        Parameters:
        -----------
        zscore_series : pd.Series
            Z-score series
        lookback_periods : int, optional
            Number of periods to look back for percentile calculation
            
        Returns:
        --------
        Dict containing percentile information
        """
        clean_zscore = zscore_series.dropna()
        
        if len(clean_zscore) == 0:
            return {'error': 'No valid z-score data'}
        
        if lookback_periods:
            reference_data = clean_zscore.tail(lookback_periods)
        else:
            reference_data = clean_zscore
        
        current_zscore = clean_zscore.iloc[-1]
        
        # Calculate percentile rank
        percentile_rank = (reference_data < current_zscore).sum() / len(reference_data) * 100
        
        return {
            'current_zscore': current_zscore,
            'percentile_rank': percentile_rank,
            'reference_periods': len(reference_data),
            'interpretation': self._interpret_percentile_rank(percentile_rank)
        }
    
    def _interpret_percentile_rank(self, percentile_rank: float) -> str:
        """Interpret percentile rank for trading context."""
        if percentile_rank >= 97.5:
            return 'extremely_high'
        elif percentile_rank >= 90:
            return 'very_high'
        elif percentile_rank >= 75:
            return 'high'
        elif percentile_rank >= 25:
            return 'normal'
        elif percentile_rank >= 10:
            return 'low'
        elif percentile_rank >= 2.5:
            return 'very_low'
        else:
            return 'extremely_low'