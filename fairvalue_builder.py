#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module C - Fair-Value Builder
Constructs fair-value price series for Asset A based on relationship with Asset B.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from regression_core import RegressionCore
from zscore_engine import ZScoreEngine
import warnings

class FairValueBuilder:
    """
    Fair-value construction module for pair trading analysis.
    
    This module constructs the fair-value price series for Asset A by applying
    the relationship discovered through regression analysis. The result is a time series
    of what Asset A "should" have been worth at each moment if it had moved perfectly
    in line with Asset B.
    """
    
    def __init__(self, regression_core: Optional[RegressionCore] = None,
                 zscore_engine: Optional[ZScoreEngine] = None):
        """
        Initialize the Fair-Value Builder.
        
        Parameters:
        -----------
        regression_core : RegressionCore, optional
            Pre-configured regression core instance
        zscore_engine : ZScoreEngine, optional
            Pre-configured z-score engine instance
        """
        self.regression_core = regression_core
        self.zscore_engine = zscore_engine
        self.fair_value_data = {}
        
    def build_fair_value_series(self, asset_a: pd.Series, asset_b: pd.Series,
                               regression_config: Optional[Dict] = None,
                               zscore_config: Optional[Dict] = None) -> Dict:
        """
        Build comprehensive fair-value analysis for a pair of assets.
        
        Parameters:
        -----------
        asset_a : pd.Series
            Target asset (dependent variable)
        asset_b : pd.Series
            Independent asset (explanatory variable)
        regression_config : Dict, optional
            Configuration for regression analysis
        zscore_config : Dict, optional
            Configuration for z-score analysis
            
        Returns:
        --------
        Dict containing complete fair-value analysis
        """
        # Set up regression core if not provided
        if self.regression_core is None:
            reg_config = regression_config or {}
            self.regression_core = RegressionCore(
                window_size=reg_config.get('window_size', 60),
                min_periods=reg_config.get('min_periods', 30),
                method=reg_config.get('method', 'ols'),
                rolling=reg_config.get('rolling_regression', True)
            )
        
        # Set up z-score engine if not provided
        if self.zscore_engine is None:
            zscore_conf = zscore_config or {}
            self.zscore_engine = ZScoreEngine(
                window_size=zscore_conf.get('window_size', 60),
                min_periods=zscore_conf.get('min_periods', 30),
                calculation_method=zscore_conf.get('calculation_method', 'rolling')
            )
        
        # Run regression analysis
        #print("[INFO] Running regression analysis...")
        regression_results = self.regression_core.fit_regression(asset_a, asset_b)
        
        # Extract key series
        fair_value_series = self.regression_core.get_fair_value_series()
        spread_series = self.regression_core.get_spread_series()
        hedge_ratio_series = self.regression_core.get_hedge_ratio_series()
        
        # Calculate z-scores for the spread
        #print("[INFO] Calculating z-scores...")
        spread_zscore = self.zscore_engine.calculate_zscore(spread_series)
        
        # Build comprehensive result
        self.fair_value_data = {
            'timestamps': fair_value_series.index,
            'asset_a_actual': asset_a.reindex(fair_value_series.index),
            'asset_b_actual': asset_b.reindex(fair_value_series.index),
            'fair_value': fair_value_series,
            'spread': spread_series,
            'spread_zscore': spread_zscore,
            'hedge_ratio': hedge_ratio_series,
            'regression_results': regression_results,
            'analysis_summary': self._create_analysis_summary(
                asset_a, asset_b, fair_value_series, spread_series, spread_zscore
            )
        }
        
        #print(f"[INFO] Fair-value analysis completed for {len(fair_value_series)} periods")
        return self.fair_value_data
    
    def _create_analysis_summary(self, asset_a: pd.Series, asset_b: pd.Series,
                               fair_value: pd.Series, spread: pd.Series, 
                               zscore: pd.Series) -> Dict:
        """Create comprehensive analysis summary."""
        
        # Align all series
        common_index = fair_value.index
        asset_a_aligned = asset_a.reindex(common_index)
        asset_b_aligned = asset_b.reindex(common_index)
        
        # Current values
        current_asset_a = asset_a_aligned.iloc[-1]
        current_asset_b = asset_b_aligned.iloc[-1]
        current_fair_value = fair_value.iloc[-1]
        current_spread = spread.iloc[-1]
        current_zscore = zscore.dropna().iloc[-1] if not zscore.dropna().empty else np.nan
        
        # Historical statistics
        spread_stats = {
            'mean': spread.mean(),
            'std': spread.std(),
            'min': spread.min(),
            'max': spread.max(),
            'current': current_spread
        }
        
        zscore_stats = {
            'mean': zscore.mean(),
            'std': zscore.std(),
            'min': zscore.min(),
            'max': zscore.max(),
            'current': current_zscore
        }
        
        # Pricing analysis
        pricing_analysis = {
            'asset_a_vs_fair_value': {
                'current_actual': current_asset_a,
                'current_fair_value': current_fair_value,
                'absolute_difference': current_spread,
                'percentage_difference': (current_spread / current_fair_value) * 100 if current_fair_value != 0 else 0,
                'overvalued': current_spread > 0
            }
        }
        
        # Correlation analysis
        correlation = asset_a_aligned.corr(asset_b_aligned)
        
        return {
            'current_values': {
                'asset_a': current_asset_a,
                'asset_b': current_asset_b,
                'fair_value': current_fair_value,
                'spread': current_spread,
                'zscore': current_zscore
            },
            'spread_statistics': spread_stats,
            'zscore_statistics': zscore_stats,
            'pricing_analysis': pricing_analysis,
            'correlation': correlation,
            'data_quality': {
                'total_periods': len(fair_value),
                'valid_zscores': zscore.dropna().count(),
                'zscore_coverage': (zscore.dropna().count() / len(fair_value)) * 100
            }
        }
    
    def get_trading_signals(self, entry_zscore: float = 2.0, 
                          exit_zscore: float = 0.5) -> Dict:
        """
        Generate trading signals based on fair-value analysis.
        
        Parameters:
        -----------
        entry_zscore : float
            Z-score threshold for entry signals
        exit_zscore : float
            Z-score threshold for exit signals
            
        Returns:
        --------
        Dict containing trading signals and analysis
        """
        if not self.fair_value_data:
            raise ValueError("No fair-value data available. Run build_fair_value_series first.")
        
        spread_zscore = self.fair_value_data['spread_zscore']
        
        # Get latest signals from z-score engine
        signals = self.zscore_engine.get_mean_reversion_signals(
            spread_zscore, entry_zscore, exit_zscore
        )
        
        # Add fair-value context
        current_data = self.fair_value_data['analysis_summary']['current_values']
        
        signals['fair_value_context'] = {
            'asset_a_actual': current_data['asset_a'],
            'asset_a_fair_value': current_data['fair_value'],
            'overvaluation': current_data['spread'],
            'overvaluation_pct': (current_data['spread'] / current_data['fair_value']) * 100 if current_data['fair_value'] != 0 else 0
        }
        
        # Add position sizing suggestions
        signals['position_sizing'] = self._calculate_position_sizing(
            signals['signal_strength'], signals['confidence']
        )
        
        return signals
    
    def _calculate_position_sizing(self, signal_strength: float, confidence: float) -> Dict:
        """Calculate suggested position sizing based on signal quality."""
        base_size = 0.1  # 10% base position
        
        # Adjust based on signal strength and confidence
        size_multiplier = min(signal_strength * confidence, 2.0)  # Cap at 2x
        suggested_size = base_size * size_multiplier
        
        return {
            'base_size_pct': base_size * 100,
            'signal_multiplier': size_multiplier,
            'suggested_size_pct': suggested_size * 100,
            'risk_assessment': 'high' if suggested_size > 0.15 else 'medium' if suggested_size > 0.08 else 'low'
        }
    
    def create_visualization_data(self, confidence_level: float = 0.95) -> Dict:
        """
        Prepare data for visualization in the main plotter.
        
        Parameters:
        -----------
        confidence_level : float
            Confidence level for bands calculation
            
        Returns:
        --------
        Dict containing all data needed for plotting
        """
        if not self.fair_value_data:
            raise ValueError("No fair-value data available. Run build_fair_value_series first.")
        
        # Get confidence bands
        try:
            lower_band, upper_band = self.regression_core.calculate_confidence_bands(confidence_level)
        except Exception as e:
            warnings.warn(f"Could not calculate confidence bands: {e}")
            lower_band = upper_band = None
        
        viz_data = {
            'timestamps': self.fair_value_data['timestamps'],
            'asset_a_actual': self.fair_value_data['asset_a_actual'],
            'asset_b_actual': self.fair_value_data['asset_b_actual'],
            'fair_value': self.fair_value_data['fair_value'],
            'spread': self.fair_value_data['spread'],
            'spread_zscore': self.fair_value_data['spread_zscore'],
            'hedge_ratio': self.fair_value_data['hedge_ratio'],
            'confidence_bands': {
                'lower': lower_band,
                'upper': upper_band,
                'confidence_level': confidence_level
            } if lower_band is not None else None,
            'summary': self.fair_value_data['analysis_summary']
        }
        
        return viz_data
    
    def export_analysis(self, filepath: str, format: str = 'csv') -> bool:
        """
        Export fair-value analysis to file.
        
        Parameters:
        -----------
        filepath : str
            Output file path
        format : str
            Export format ('csv', 'json', 'excel')
            
        Returns:
        --------
        bool indicating success
        """
        if not self.fair_value_data:
            print("[ERROR] No fair-value data available for export")
            return False
        
        try:
            # Prepare export data
            export_df = pd.DataFrame({
                'timestamp': self.fair_value_data['timestamps'],
                'asset_a_actual': self.fair_value_data['asset_a_actual'],
                'asset_b_actual': self.fair_value_data['asset_b_actual'],
                'fair_value': self.fair_value_data['fair_value'],
                'spread': self.fair_value_data['spread'],
                'spread_zscore': self.fair_value_data['spread_zscore'],
                'hedge_ratio': self.fair_value_data['hedge_ratio']
            })
            
            if format.lower() == 'csv':
                export_df.to_csv(filepath, index=False)
            elif format.lower() == 'json':
                # Include summary in JSON export
                export_data = {
                    'data': export_df.to_dict('records'),
                    'summary': self.fair_value_data['analysis_summary']
                }
                import json
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            elif format.lower() == 'excel':
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    export_df.to_excel(writer, sheet_name='Fair_Value_Data', index=False)
                    
                    # Add summary sheet
                    summary_df = pd.DataFrame([self.fair_value_data['analysis_summary']['current_values']])
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
            else:
                #print(f"[ERROR] Unsupported export format: {format}")
                return False
            
            #print(f"[INFO] Analysis exported to {filepath}")
            return True
            
        except Exception as e:
            #print(f"[ERROR] Export failed: {e}")
            return False
    
    def get_performance_metrics(self, lookback_periods: Optional[int] = None) -> Dict:
        """
        Calculate performance metrics for the fair-value model.
        
        Parameters:
        -----------
        lookback_periods : int, optional
            Number of periods to analyze (None for all data)
            
        Returns:
        --------
        Dict containing performance metrics
        """
        if not self.fair_value_data:
            raise ValueError("No fair-value data available")
        
        spread = self.fair_value_data['spread']
        zscore = self.fair_value_data['spread_zscore'].dropna()
        
        if lookback_periods:
            spread = spread.tail(lookback_periods)
            zscore = zscore.tail(lookback_periods)
        
        # Calculate various metrics
        metrics = {
            'mean_reversion_strength': self._calculate_mean_reversion_strength(spread),
            'zscore_distribution': {
                'mean': zscore.mean(),
                'std': zscore.std(),
                'skewness': zscore.skew(),
                'kurtosis': zscore.kurtosis()
            },
            'model_quality': {
                'avg_r_squared': self.fair_value_data['regression_results']['summary_stats'].get('avg_r_squared', 0),
                'spread_volatility': spread.std(),
                'hedge_ratio_stability': self.fair_value_data['hedge_ratio'].std()
            },
            'trading_opportunities': self._count_trading_opportunities(zscore)
        }
        
        return metrics
    
    def _calculate_mean_reversion_strength(self, spread: pd.Series) -> float:
        """Calculate the mean reversion strength of the spread."""
        # Use autocorrelation at lag 1 as a measure of mean reversion
        # Values closer to 0 indicate stronger mean reversion
        try:
            autocorr = spread.autocorr(lag=1)
            return 1 - abs(autocorr) if not np.isnan(autocorr) else 0.0
        except:
            return 0.0
    
    def _count_trading_opportunities(self, zscore: pd.Series, threshold: float = 2.0) -> Dict:
        """Count potential trading opportunities based on z-score thresholds."""
        long_signals = (zscore < -threshold).sum()
        short_signals = (zscore > threshold).sum()
        neutral_periods = (abs(zscore) <= 0.5).sum()
        
        return {
            'long_entry_signals': long_signals,
            'short_entry_signals': short_signals,
            'total_entry_signals': long_signals + short_signals,
            'neutral_periods': neutral_periods,
            'signal_frequency_pct': ((long_signals + short_signals) / len(zscore)) * 100
        }