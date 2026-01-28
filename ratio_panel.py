#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ratio Panel Implementation
Provides price ratio analysis with volume comparison
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from matplotlib.axes import Axes

from panel_base import PanelBase, PanelData, PanelResult, ThemeConfig

class RatioPanel(PanelBase):
    """Panel for price ratio analysis with volume comparison"""
    
    def get_required_data_fields(self) -> List[str]:
        """Return required fields for ratio analysis"""
        return ['close', 'volume']
    
    def plot(self, ax: Axes, ax_secondary: Optional[Axes], 
             data: PanelData, panel_idx: int, panel_count: int) -> PanelResult:
        """Plot price ratio panel with volume comparison"""
        
        # Validate data
        is_valid, error_msg = self.validate_data(data)
        if not is_valid:
            return PanelResult(success=False, error_message=error_msg)
        
        if data.df2 is None:
            return PanelResult(success=False, error_message="Second asset data required for ratio analysis")
        
        print(f"\n=== PANEL 1: {data.asset1}/{data.asset2} RATIO CHART WITH VOLUME ===")
        
        try:
            # Calculate ratio and volume ratio
            ratio = data.df1['close'] / data.df2['close']
            volume_ratio = data.df1['volume'] / data.df2['volume']
            volume_ratio = volume_ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Plot price ratio as line
            ax.plot(ratio.index, ratio, color=self.theme.primary_color, 
                   linewidth=1.5, alpha=0.9, label=f'{data.asset1}/{data.asset2} Ratio')
            
            # Plot volume ratio bars on secondary axis
            if ax_secondary is not None:
                ax_secondary.bar(data.df1.index, volume_ratio, alpha=0.4, 
                               color='#00b4d8', width=data.volume_bar_width, 
                               label='Volume Ratio')
                
                # Scale volume to bottom 20% of chart
                vol_max = volume_ratio.max() if volume_ratio.max() > 0 else 1
                ax_secondary.set_ylim(0, vol_max * 5)
                ax_secondary.set_ylabel('Vol Ratio', color='#00b4d8', fontsize=9)
                ax_secondary.tick_params(axis='y', labelcolor='#00b4d8', labelsize=8)
                ax_secondary.yaxis.set_label_position('right')
                ax_secondary.yaxis.tick_right()
            
            # Calculate statistics
            ratio_stats = self.calculate_basic_statistics(ratio, 'ratio')
            volume_stats = self.calculate_basic_statistics(volume_ratio, 'volume_ratio')
            
            statistics = {**ratio_stats, **volume_stats}
            
            # Print statistics
            print(f"Price Ratio:")
            print(f"  Min:     {statistics['ratio_min']:.6f}")
            print(f"  Max:     {statistics['ratio_max']:.6f}")
            print(f"  Latest:  {statistics['ratio_latest']:.6f}")
            print(f"Volume Ratio:")
            print(f"  Average: {statistics['volume_ratio_mean']:.4f}")
            print(f"  Latest:  {statistics['volume_ratio_latest']:.4f}")
            
            # Set y-axis limits with padding
            ratio_padding = (statistics['ratio_max'] - statistics['ratio_min']) * 0.05
            y_limits = (statistics['ratio_min'] - ratio_padding, 
                       statistics['ratio_max'] + ratio_padding)
            ax.set_ylim(*y_limits)
            
            # Add text annotation
            stats_text = (f'Price Ratio: {statistics["ratio_latest"]:.6f}\n'
                         f'Min: {statistics["ratio_min"]:.6f}\n'
                         f'Max: {statistics["ratio_max"]:.6f}\n'
                         f'Vol Ratio: {statistics["volume_ratio_latest"]:.2f}')
            
            self.add_text_annotation(ax, stats_text, (0.25, 0.85), self.theme.primary_color)
            
            # Set title
            title = f'{data.asset1}/{data.asset2} • {data.timeframe.upper()} • Ratio Chart with Volume'
            
            # Apply common styling
            self.apply_common_styling(ax, title, 'Price Ratio')
            
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
            error_msg = f"Ratio panel plotting failed: {e}"
            print(f"[ERROR] {error_msg}")
            return PanelResult(success=False, error_message=error_msg)