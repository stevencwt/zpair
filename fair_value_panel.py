#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fair Value Panel Implementation - FIXED VERSION
Provides fair-value analysis with regression-based spread calculation
FIXED: Spread bar width calculation for high-frequency data
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from matplotlib.axes import Axes

from panel_base import PanelBase, PanelData, PanelResult, ThemeConfig

class FairValuePanel(PanelBase):
    """Panel for fair-value analysis using regression and spread calculation"""
    
    def __init__(self, panel_config: Dict, theme_config: ThemeConfig):
        super().__init__(panel_config, theme_config)
        self.modules_available = self._check_modules()
    
    def _check_modules(self) -> bool:
        """Check if required modules are available"""
        try:
            from fairvalue_builder import FairValueBuilder
            return True
        except ImportError:
            return False
    
    def get_required_data_fields(self) -> List[str]:
        """Return required fields for fair value analysis"""
        return ['close', 'volume']
    
    def _calculate_optimal_spread_bar_width(self, data: PanelData) -> float:
        """Calculate optimal bar width specifically for spread/differential display"""
        if len(data.df1) < 2:
            return 0.0001  # Very narrow default for spread bars
        
        # Calculate actual time intervals between data points
        time_diffs = []
        for i in range(1, min(len(data.df1), 20)):  # Sample last 20 points
            diff = (data.df1.index[i] - data.df1.index[i-1]).total_seconds()
            if diff > 0:
                time_diffs.append(diff)
        
        if time_diffs:
            # Use 60% of average interval for spread bars (narrower than volume)
            avg_interval = sum(time_diffs) / len(time_diffs)
            optimal_width = (avg_interval * 0.6) / 86400  # Convert to days
            
            # Apply bounds based on data frequency
            if avg_interval <= 60:  # Sub-minute data
                optimal_width = max(optimal_width, 0.00001)  # Very narrow
                optimal_width = min(optimal_width, 0.0002)   # Max for sub-minute
            elif avg_interval <= 300:  # 5-minute data
                optimal_width = max(optimal_width, 0.0001)
                optimal_width = min(optimal_width, 0.001)
            else:  # Hourly+ data
                optimal_width = max(optimal_width, 0.001)
                optimal_width = min(optimal_width, 0.01)
            
            #print(f"[SPREAD DEBUG] Avg interval: {avg_interval:.1f}s, Bar width: {optimal_width:.8f}")
            return optimal_width
        
        # Fallback based on timeframe
        timeframe_widths = {
            '1m': 0.00003,   # ~2.5 seconds
            '5m': 0.0002,    # ~17 seconds  
            '15m': 0.0007,   # ~1 minute
            '1h': 0.002,     # ~3 minutes
            '4h': 0.008,     # ~11 minutes
            '1d': 0.2
        }
        return timeframe_widths.get(data.timeframe, 0.0002)
    
    def plot(self, ax: Axes, ax_secondary: Optional[Axes], 
             data: PanelData, panel_idx: int, panel_count: int) -> PanelResult:
        """Plot fair-value analysis panel with spread display"""
        
        # Validate modules availability
        if not self.modules_available:
            #print("[WARN] Fair-value modules not available, falling back to ratio chart")
            return self._fallback_to_ratio(ax, ax_secondary, data, panel_idx, panel_count)
        
        # Validate data
        is_valid, error_msg = self.validate_data(data)
        if not is_valid:
            return PanelResult(success=False, error_message=error_msg)
        
        if data.df2 is None:
            return PanelResult(success=False, error_message="Second asset data required for fair value analysis")
        
        #print(f"\n=== PANEL 1: {data.asset1} FAIR VALUE ANALYSIS ===")
        
        try:
            # Import modules
            from fairvalue_builder import FairValueBuilder
            
            # Get configuration settings
            fair_value_options = self.config.get('fair_value', {})
            regression_settings = self.config.get('regression_settings', {})
            zscore_settings = self.config.get('zscore_settings', {})
            
            # Initialize the fair-value builder
            regression_config = {
                'window_size': regression_settings.get('window_size', 60),
                'min_periods': regression_settings.get('min_periods', 30),
                'rolling_regression': regression_settings.get('rolling_regression', True)
            }
            
            zscore_config = {
                'window_size': zscore_settings.get('window_size', 60),
                'min_periods': zscore_settings.get('min_periods', 30),
                'calculation_method': zscore_settings.get('calculation_method', 'rolling')
            }
            
            builder = FairValueBuilder()
            
            # DEBUG: Show input data for comparison with dashboard
            #print(f"[CHART DEBUG] Input data for regression:")
            #print(f"  {data.asset1} latest: {data.df1['close'].iloc[-1]:.6f} at {data.df1.index[-1]}")
            #print(f"  {data.asset2} latest: {data.df2['close'].iloc[-1]:.6f} at {data.df2.index[-1]}")
            #print(f"  {data.asset1} data length: {len(data.df1)}")
            #print(f"  {data.asset2} data length: {len(data.df2)}")
            #print(f"  Regression window: {regression_config['window_size']}")
            #print(f"  Z-score window: {zscore_config['window_size']}")

            # NEW: Add detailed debugging before FairValueBuilder
            #print(f"[DEBUG] Data validation before FairValueBuilder:")
            #print(f"  df1 shape: {data.df1.shape}, df2 shape: {data.df2.shape}")
            #print(f"  df1.index type: {type(data.df1.index)}, df2.index type: {type(data.df2.index)}")
            #print(f"  df1 close dtype: {data.df1['close'].dtype}, df2 close dtype: {data.df2['close'].dtype}")
            #print(f"  df1 close nulls: {data.df1['close'].isnull().sum()}, df2 close nulls: {data.df2['close'].isnull().sum()}")
            #print(f"  Index alignment check: {data.df1.index.equals(data.df2.index)}")
            #print(f"  df1 close sample: {data.df1['close'].tail(3).tolist()}")
            #print(f"  df2 close sample: {data.df2['close'].tail(3).tolist()}")

            # Build fair-value analysis with enhanced error handling
            try:
                   fair_value_data = builder.build_fair_value_series(
                      asset_a=data.df1['close'], 
                      asset_b=data.df2['close'],
                      regression_config=regression_config,
                      zscore_config=zscore_config
                   )
            except Exception as fv_error:
                   print(f"[ERROR] FairValueBuilder failed with: {fv_error}")
                   print(f"[ERROR] Exception type: {type(fv_error)}")
                   import traceback
                   traceback.print_exc()
                   print(f"[FALLBACK] Falling back to ratio chart due to FairValueBuilder error")
                   return self._fallback_to_ratio(ax, ax_secondary, data, panel_idx, panel_count)

                        
            # Extract visualization data
            confidence_level = fair_value_options.get('confidence_level', 0.95)
            viz_data = builder.create_visualization_data(confidence_level=confidence_level)
            
            #print(f"Visualization data lengths:")
            #print(f"  Timestamps: {len(viz_data['timestamps'])}")
            #print(f"  Actual values: {len(viz_data['asset_a_actual'])}")
            #print(f"  Fair values: {len(viz_data['fair_value'])}")
            
            # Check data validity
            actual_valid = viz_data['asset_a_actual'].dropna()
            fair_valid = viz_data['fair_value'].dropna()
            
            if len(actual_valid) == 0 or len(fair_valid) == 0:
                #print("[WARN] Insufficient valid data for fair-value analysis, falling back to ratio")
                return self._fallback_to_ratio(ax, ax_secondary, data, panel_idx, panel_count)
            
            # Plot actual vs fair value
            ax.plot(viz_data['timestamps'], viz_data['asset_a_actual'], 
                   color=self.theme.primary_color, linewidth=2.0, alpha=0.9, 
                   label=f'{data.asset1} Actual', linestyle='-')
            
            ax.plot(viz_data['timestamps'], viz_data['fair_value'], 
                   color=self.theme.success_color, linewidth=2.0, alpha=0.8, 
                   label=f'{data.asset1} Fair Value', linestyle='--')
            
            #print(f"Plotted lines:")
            #print(f"  Actual line: {len(viz_data['asset_a_actual'])} points")
            #print(f"  Fair value line: {len(viz_data['fair_value'])} points")
            
            # Add confidence bands if available and requested
            if (viz_data['confidence_bands'] and 
                fair_value_options.get('show_confidence_bands', True)):
                lower_band = viz_data['confidence_bands']['lower']
                upper_band = viz_data['confidence_bands']['upper']
                
                if lower_band is not None and upper_band is not None:
                    ax.fill_between(viz_data['timestamps'], lower_band, upper_band, 
                                   alpha=0.15, color=self.theme.success_color, 
                                   label=f'{confidence_level*100:.0f}% Confidence')
            
            # Display spread (residuals) on secondary axis with FIXED bar width
            if ax_secondary is not None:
                spread_color = '#ff6b35'
                spread_data = viz_data['spread'].dropna()
                
                if len(spread_data) > 0:
                    # Calculate optimal bar width for spread display
                    spread_bar_width = self._calculate_optimal_spread_bar_width(data)
                    
                    #print(f"[SPREAD DEBUG] Using spread bar width: {spread_bar_width:.8f}")
                    #print(f"[SPREAD DEBUG] Spread data: {len(spread_data)} points")
                    #print(f"[SPREAD DEBUG] Spread range: {spread_data.min():.4f} to {spread_data.max():.4f}")
                    
                    ax_secondary.bar(spread_data.index, spread_data.values, 
                                   alpha=0.6, color=spread_color, width=spread_bar_width, 
                                   label='Spread (Actual - Fair)')
                    
                    # Scale spread appropriately
                    spread_max = abs(spread_data).max()
                    spread_range = spread_max * 2.5
                    ax_secondary.set_ylim(-spread_range, spread_range)
                    ax_secondary.set_ylabel('Spread', color=spread_color, fontsize=9)
                    ax_secondary.tick_params(axis='y', labelcolor=spread_color, labelsize=8)
                    
                    # Add zero line for spread reference
                    ax_secondary.axhline(y=0, color=spread_color, linestyle='-', linewidth=0.5, alpha=0.7)
                    
                    #print(f"Spread data: {len(spread_data)} points, range: ±{spread_max:.4f}")
                
                ax_secondary.yaxis.set_label_position('right')
                ax_secondary.yaxis.tick_right()
            
            # Calculate and collect statistics
            summary = viz_data['summary']
            current = summary['current_values']
            pricing = summary['pricing_analysis']['asset_a_vs_fair_value']
            
            statistics = {
                'actual_price': current['asset_a'],
                'fair_value_price': current['fair_value'],
                'spread': current['spread'],
                'spread_percentage': pricing['percentage_difference'],
                'is_overvalued': pricing['overvalued'],
                'zscore': current['zscore'],
                'correlation': summary['correlation']
            }
            
            # Get regression stats if available
            reg_results = summary.get('regression_results', {})
            latest_params = reg_results.get('latest_params', {})
            if latest_params:
                statistics.update({
                    'r_squared': latest_params.get('r_squared', 0),
                    'slope': latest_params.get('slope', 0),
                    'intercept': latest_params.get('intercept', 0)
                })
            
            # Print statistics
            #print(f"Fair Value Analysis:")
            #print(f"  {data.asset1} Actual:     {current['asset_a']:.6f}")
            #print(f"  {data.asset1} Fair Value: {current['fair_value']:.6f}")
            #print(f"  Spread:             {current['spread']:.6f}")
            #print(f"  Spread %:           {pricing['percentage_difference']:.2f}%")
            #print(f"  Status:             {'OVERVALUED' if pricing['overvalued'] else 'UNDERVALUED'}")
            #print(f"  Z-Score:            {current['zscore']:.4f}")
            
            #if latest_params:
                #print(f"Regression Stats:")
                #print(f"  R-squared:          {latest_params.get('r_squared', 'N/A'):.4f}")
                #print(f"  Slope (Hedge):      {latest_params.get('slope', 'N/A'):.4f}")
                #print(f"  Intercept:          {latest_params.get('intercept', 'N/A'):.4f}")
                #print(f"  Correlation:        {summary['correlation']:.4f}")
            
            # Set appropriate y-axis limits
            actual_min, actual_max = viz_data['asset_a_actual'].min(), viz_data['asset_a_actual'].max()
            fair_min, fair_max = viz_data['fair_value'].min(), viz_data['fair_value'].max()
            
            combined_min = min(actual_min, fair_min)
            combined_max = max(actual_max, fair_max)
            y_padding = (combined_max - combined_min) * 0.05
            y_limits = (combined_min - y_padding, combined_max + y_padding)
            ax.set_ylim(*y_limits)
            
            #print(f"Y-axis range: {y_limits[0]:.4f} to {y_limits[1]:.4f}")
            
            # Create status text annotation
            status_text = 'OVERVALUED' if pricing['overvalued'] else 'UNDERVALUED'
            stats_text = (f'Actual: {current["asset_a"]:.4f}\n'
                         f'Fair: {current["fair_value"]:.4f}\n'
                         f'Spread: {current["spread"]:.4f}\n'
                         f'Status: {status_text}\n'
                         f'Z-Score: {current["zscore"]:.2f}')
            
            self.add_text_annotation(ax, stats_text, (0.25, 0.85), self.theme.success_color)
            
            # Set title
            confidence_text = f" (Conf: {confidence_level*100:.0f}%)" if fair_value_options.get('show_confidence_bands') else ""
            title = f'{data.asset1} Fair Value vs {data.asset2} • {data.timeframe.upper()}{confidence_text}'
            
            # Apply common styling
            self.apply_common_styling(ax, title, f'{data.asset1} Price')
            
            # Format x-axis if last panel
            self.format_xaxis_if_last_panel(ax, data, panel_idx, panel_count)
            
            # Create legend
            ax.legend(loc='upper left', frameon=False, fontsize=10)
            
            return PanelResult(
                statistics=statistics,
                y_limits=y_limits,
                title=title,
                success=True
            )
            
        except Exception as e:
            print(f"[ERROR] Fair-value analysis failed: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to ratio chart
            return self._fallback_to_ratio(ax, ax_secondary, data, panel_idx, panel_count)
    
    def _fallback_to_ratio(self, ax: Axes, ax_secondary: Optional[Axes], 
                          data: PanelData, panel_idx: int, panel_count: int) -> PanelResult:
        """Fallback to ratio chart when fair-value analysis fails"""
        from ratio_panel import RatioPanel
        
        ratio_panel = RatioPanel(self.config, self.theme)
        return ratio_panel.plot(ax, ax_secondary, data, panel_idx, panel_count)
