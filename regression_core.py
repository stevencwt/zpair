#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module A - Regression Core
Studies the historical relationship between two assets and generates fair-value estimates.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings

class RegressionCore:
    """
    Core regression module for pair trading analysis.
    
    This module studies the historical relationship between two assets (Asset A and Asset B).
    It estimates how much Asset A's price is typically explained by Asset B's price,
    capturing both a base offset (intercept) and a proportional movement (slope/hedge ratio).
    """
    
    def __init__(self, window_size: int = 60, min_periods: int = 30, 
                 method: str = 'ols', rolling: bool = True):
        """
        Initialize the regression core.
        
        Parameters:
        -----------
        window_size : int
            Rolling window size for regression calculation
        min_periods : int
            Minimum number of periods required for calculation
        method : str
            Regression method ('ols' for Ordinary Least Squares)
        rolling : bool
            Whether to use rolling regression or expanding window
        """
        self.window_size = window_size
        self.min_periods = min_periods
        self.method = method
        self.rolling = rolling
        self.regression_results = {}
        
    def fit_regression(self, asset_a: pd.Series, asset_b: pd.Series) -> Dict:
        """
        Fit regression model between two assets.
        
        Parameters:
        -----------
        asset_a : pd.Series
            Target asset (dependent variable)
        asset_b : pd.Series
            Independent asset (explanatory variable)
            
        Returns:
        --------
        Dict containing regression results
        """
        # Align series on common index
        aligned_data = pd.concat([asset_a, asset_b], axis=1, join='inner')
        aligned_data.columns = ['asset_a', 'asset_b']
        aligned_data = aligned_data.dropna()
        
        if len(aligned_data) < self.min_periods:
            raise ValueError(f"Insufficient data: {len(aligned_data)} < {self.min_periods}")
        
        results = {
            'intercept': [],
            'slope': [],
            'r_squared': [],
            'fair_value': [],
            'spread': [],
            'timestamps': [],
            'std_error': []
        }
        
        if self.rolling:
            # Rolling regression
            for i in range(self.min_periods - 1, len(aligned_data)):
                start_idx = max(0, i - self.window_size + 1)
                window_data = aligned_data.iloc[start_idx:i+1]
                
                reg_result = self._calculate_single_regression(
                    window_data['asset_a'], window_data['asset_b']
                )
                
                # Calculate fair value for current point
                current_asset_b = aligned_data.iloc[i]['asset_b']
                fair_value = reg_result['intercept'] + reg_result['slope'] * current_asset_b
                actual_value = aligned_data.iloc[i]['asset_a']
                spread = actual_value - fair_value
                
                results['intercept'].append(reg_result['intercept'])
                results['slope'].append(reg_result['slope'])
                results['r_squared'].append(reg_result['r_squared'])
                results['fair_value'].append(fair_value)
                results['spread'].append(spread)
                results['std_error'].append(reg_result['std_error'])
                results['timestamps'].append(aligned_data.index[i])
        else:
            # Expanding window regression
            for i in range(self.min_periods - 1, len(aligned_data)):
                window_data = aligned_data.iloc[:i+1]
                
                reg_result = self._calculate_single_regression(
                    window_data['asset_a'], window_data['asset_b']
                )
                
                # Calculate fair value for current point
                current_asset_b = aligned_data.iloc[i]['asset_b']
                fair_value = reg_result['intercept'] + reg_result['slope'] * current_asset_b
                actual_value = aligned_data.iloc[i]['asset_a']
                spread = actual_value - fair_value
                
                results['intercept'].append(reg_result['intercept'])
                results['slope'].append(reg_result['slope'])
                results['r_squared'].append(reg_result['r_squared'])
                results['fair_value'].append(fair_value)
                results['spread'].append(spread)
                results['std_error'].append(reg_result['std_error'])
                results['timestamps'].append(aligned_data.index[i])
        
        # Convert to DataFrame for easier handling
        self.regression_results = pd.DataFrame(results)
        self.regression_results.set_index('timestamps', inplace=True)
        
        return {
            'regression_data': self.regression_results,
            'summary_stats': self._calculate_summary_stats(),
            'latest_params': {
                'intercept': results['intercept'][-1],
                'slope': results['slope'][-1],
                'r_squared': results['r_squared'][-1],
                'std_error': results['std_error'][-1]
            }
        }
    
    def _calculate_single_regression(self, y: pd.Series, x: pd.Series) -> Dict:
        """Calculate single regression between two series."""
        try:
            # Remove any remaining NaN values
            clean_data = pd.concat([y, x], axis=1).dropna()
            if len(clean_data) < 2:
                return {'intercept': 0, 'slope': 1, 'r_squared': 0, 'std_error': np.inf}
            
            y_clean = clean_data.iloc[:, 0].values.reshape(-1, 1)
            x_clean = clean_data.iloc[:, 1].values.reshape(-1, 1)
            
            # Fit linear regression
            model = LinearRegression()
            model.fit(x_clean, y_clean)
            
            # Calculate predictions and R²
            y_pred = model.predict(x_clean)
            r_squared = r2_score(y_clean, y_pred)
            
            # Calculate standard error
            residuals = y_clean.flatten() - y_pred.flatten()
            mse = np.mean(residuals ** 2)
            std_error = np.sqrt(mse)
            
            return {
                'intercept': model.intercept_[0],
                'slope': model.coef_[0][0],
                'r_squared': max(0, r_squared),  # Ensure non-negative
                'std_error': std_error
            }
            
        except Exception as e:
            warnings.warn(f"Regression calculation failed: {e}")
            return {'intercept': 0, 'slope': 1, 'r_squared': 0, 'std_error': np.inf}
    
    def _calculate_summary_stats(self) -> Dict:
        """Calculate summary statistics for the regression results."""
        if self.regression_results.empty:
            return {}
        
        return {
            'avg_intercept': self.regression_results['intercept'].mean(),
            'avg_slope': self.regression_results['slope'].mean(),
            'avg_r_squared': self.regression_results['r_squared'].mean(),
            'slope_volatility': self.regression_results['slope'].std(),
            'intercept_volatility': self.regression_results['intercept'].std(),
            'spread_mean': self.regression_results['spread'].mean(),
            'spread_std': self.regression_results['spread'].std(),
            'min_r_squared': self.regression_results['r_squared'].min(),
            'max_r_squared': self.regression_results['r_squared'].max()
        }
    
    def get_fair_value_series(self) -> pd.Series:
        """Return the fair value series."""
        if self.regression_results.empty:
            raise ValueError("No regression results available. Run fit_regression first.")
        return self.regression_results['fair_value']
    
    def get_spread_series(self) -> pd.Series:
        """Return the spread (residuals) series."""
        if self.regression_results.empty:
            raise ValueError("No regression results available. Run fit_regression first.")
        return self.regression_results['spread']
    
    def get_hedge_ratio_series(self) -> pd.Series:
        """Return the hedge ratio (slope) series."""
        if self.regression_results.empty:
            raise ValueError("No regression results available. Run fit_regression first.")
        return self.regression_results['slope']
    
    def calculate_confidence_bands(self, confidence_level: float = 0.95) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate confidence bands around fair value.
        
        Parameters:
        -----------
        confidence_level : float
            Confidence level (e.g., 0.95 for 95% confidence)
            
        Returns:
        --------
        Tuple of (lower_band, upper_band) Series
        """
        if self.regression_results.empty:
            raise ValueError("No regression results available. Run fit_regression first.")
        
        from scipy.stats import t
        
        # Calculate t-critical value
        alpha = 1 - confidence_level
        degrees_freedom = self.window_size - 2
        t_critical = t.ppf(1 - alpha/2, degrees_freedom)
        
        # Calculate confidence bands
        margin_error = t_critical * self.regression_results['std_error']
        fair_value = self.regression_results['fair_value']
        
        lower_band = fair_value - margin_error
        upper_band = fair_value + margin_error
        
        return lower_band, upper_band
    
    def get_latest_relationship(self) -> Dict:
        """Get the most recent relationship parameters."""
        if self.regression_results.empty:
            return {}
        
        latest = self.regression_results.iloc[-1]
        return {
            'intercept': latest['intercept'],
            'slope': latest['slope'],
            'r_squared': latest['r_squared'],
            'std_error': latest['std_error'],
            'timestamp': latest.name
        }
    
    def predict_fair_value(self, asset_b_value: float) -> float:
        """
        Predict fair value of asset A given asset B value using latest relationship.
        
        Parameters:
        -----------
        asset_b_value : float
            Current value of asset B
            
        Returns:
        --------
        Predicted fair value of asset A
        """
        latest = self.get_latest_relationship()
        if not latest:
            raise ValueError("No regression results available")
        
        return latest['intercept'] + latest['slope'] * asset_b_value