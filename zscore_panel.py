#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Z-Score Panel Implementation
Provides z-score analysis with threshold levels
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from matplotlib.axes import Axes

from panel_base import PanelBase, PanelData, PanelResult, ThemeConfig

class ZScorePanel(PanelBase):
    """Panel for z-score analysis with threshold visualization"""
    
    def __init__(self, panel_config: Dict, theme_config: ThemeConfig):
        super().__init__(panel_config, theme_config)
        self.zscore_engine = None
        self._try_init_zscore_engine()
    
    def _try_init_zscore_engine(self):
        """Try to initialize Z-Score engine if available"""
        try:
            from zscore_engine import ZScoreEngine
            zscore_settings = self.config.get('zscore_settings', {})
            self.zscore_engine = ZScoreEngine(
                window_size=zscore_settings.get('window_size', 60),
                min_periods=zscore_settings.get('min_periods', 30),
                calculation_method=zscore_settings.get('calculation_method', 'rolling')
            )
        except ImportError:
            self.zscore_engine = None
    
    def get_required_data_fields(self) -> List[str]:
        """Return required fields for z-score analysis"""
        return ['close']
    
    def calculate_zscore(self, ratio: pd.Series, window: int) -> pd.Series:
        """Calculate rolling z-score using ZScoreEngine if available, fallback otherwise"""
        if self.zscore_engine is not None:
            return self.zscore_engine.calculate_zscore_from_ratio(ratio)
        else:
            # Original implementation as fallback
            rolling_mean = ratio.rolling(window=window).mean()
            rolling_std = ratio.rolling(window=window).std()
            zscore = (ratio - rolling_mean) / rolling_std
            return zscore
    
    def plot(self, ax: Axes, ax_secondary: Optional[Axes], 
             data: PanelData, panel_idx: int, panel_count: int) -> PanelResult:
        """Plot z-score analysis panel"""
        
        # Validate data
        is_valid, error_msg = self.validate_data(data)
        if not is_valid:
            return PanelResult(success=False, error_message=error_msg)
        
        if data.df2 is None:
            return PanelResult(success=False, error_message="Second asset data required for z-score analysis")
        
        print(f"\n=== PANEL 2: Z-SCORE CHART (Window={data.zscore_window}) ===")
        
        try:
            # Calculate ratio and z-score
            ratio = data.df1['close'] / data.df2['close']
            zscore = self.calculate_zscore(ratio, data.zscore_window)
            
            # Plot z-score line
            ax.plot(zscore.index, zscore, color=self.theme.primary_color, 
                   linewidth=1.5, alpha=0.9, label='Z-Score')
            
            # Calculate statistics
            zscore_clean = zscore.dropna()
            statistics = {}
            
            if len(zscore_clean) > 0:
                zscore_stats = self.calculate_basic_statistics(zscore_clean, 'zscore')
                statistics.update(zscore_stats)
                
                #print(f"Min Z-Score:    {zscore_stats['zscore_min']:.4f}")
                #print(f"Max Z-Score:    {zscore_stats['zscore_max']:.4f}")
                #print(f"Latest Z-Score: {zscore_stats['zscore_latest']:.4f}")
            
            # Add reference lines
            ax.axhline(y=0, color='#787b86', linestyle='-', linewidth=1, alpha=0.5)
            
            # Add threshold lines
            colors = [self.theme.success_color, self.theme.warning_color, self.theme.danger_color]
            for i, thresh in enumerate(data.zscore_thresholds):
                color = colors[min(i, len(colors) - 1)]
                ax.axhline(y=thresh, color=color, linestyle='--', linewidth=1, 
                          alpha=0.7, label=f'+{thresh}σ')
                ax.axhline(y=-thresh, color=color, linestyle='--', linewidth=1, 
                          alpha=0.7, label=f'-{thresh}σ')
            
            # Add fill areas for positive and negative z-scores
            ax.fill_between(zscore.index, 0, zscore, where=(zscore > 0), 
                           alpha=0.1, color=self.theme.success_color, interpolate=True)
            ax.fill_between(zscore.index, 0, zscore, where=(zscore < 0), 
                           alpha=0.1, color=self.theme.danger_color, interpolate=True)
            
            # Add text annotation
            if len(zscore_clean) > 0:
                zscore_stats_text = (f'Min: {statistics["zscore_min"]:.4f}\n'
                                   f'Max: {statistics["zscore_max"]:.4f}\n'
                                   f'Latest: {statistics["zscore_latest"]:.4f}')
                
                self.add_text_annotation(ax, zscore_stats_text, (0.25, 0.85), 
                                       self.theme.primary_color)
            
            # Set title
            title = f'Z-Score (Window={data.zscore_window})'
            
            # Apply common styling
            self.apply_common_styling(ax, title, 'Z-Score')
            
            # Set y-axis limits
            ax.set_ylim(-4, 4)
            
            # Format x-axis if last panel
            self.format_xaxis_if_last_panel(ax, data, panel_idx, panel_count)
            
            # Create legend with threshold labels
            ax.legend(loc='upper left', frameon=False, fontsize=9, ncol=3)
            
            return PanelResult(
                statistics=statistics,
                y_limits=(-4, 4),
                title=title,
                success=True
            )
            
        except Exception as e:
            error_msg = f"Z-score panel plotting failed: {e}"
            #print(f"[ERROR] {error_msg}")
            return PanelResult(success=False, error_message=error_msg)