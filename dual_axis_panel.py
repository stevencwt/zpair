#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dual Axis Panel Implementation - Pure Price Comparison
Provides dual y-axis asset comparison focused on price data only
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from panel_base import PanelBase, PanelData, PanelResult, ThemeConfig

class DualAxisPanel(PanelBase):
    """Panel for dual y-axis asset comparison focusing on price data"""
    
    def __init__(self, panel_config: Dict, theme_config: ThemeConfig):
        super().__init__(panel_config, theme_config)
        self.chart_style = panel_config.get('chart_style', 'line')
        self.realtime_mode = panel_config.get('realtime_mode', False)
        self.data_frequency_seconds = panel_config.get('data_frequency_seconds', 60)
    
    def get_required_data_fields(self) -> List[str]:
        """Return required fields for dual axis display"""
        return ['open', 'high', 'low', 'close']
    
    def plot(self, ax: Axes, ax_secondary: Optional[Axes], 
             data: PanelData, panel_idx: int, panel_count: int) -> PanelResult:
        """Plot dual y-axis asset chart"""
        
        # Validate data
        is_valid, error_msg = self.validate_data(data)
        if not is_valid:
            return PanelResult(success=False, error_message=error_msg)
        
        if data.df2 is None:
            return PanelResult(success=False, error_message="Second asset data required for dual axis display")
        
        #print(f"\n=== PANEL 3: DUAL AXIS PRICE COMPARISON ===")
        
        try:
            # Create right axis for second asset if not provided
            if ax_secondary is None:
                ax_secondary = ax.twinx()
            
            # Calculate price limits
            asset1_limits = self._calculate_price_limits(data.df1)
            asset2_limits = self._calculate_price_limits(data.df2)
            
            # Set y-axis limits for both assets
            ax.set_ylim(*asset1_limits)
            ax_secondary.set_ylim(*asset2_limits)
            
            # Plot price data based on chart style
            if self.chart_style == 'candlestick':
                self._plot_candlesticks(ax, ax_secondary, data)
            else:
                self._plot_lines(ax, ax_secondary, data)
            
            # Calculate current asset values and statistics
            asset1_current = data.df1['close'].iloc[-1]
            asset2_current = data.df2['close'].iloc[-1]
            asset1_timestamp = data.df1.index[-1]
            
            statistics = {
                'asset1_current': asset1_current,
                'asset2_current': asset2_current,
                'timestamp': asset1_timestamp,
                'price_ratio': asset1_current / asset2_current
            }
            
            # Print statistics
            #print(f"{data.asset1} Current Price: {asset1_current:.6f}")
            #print(f"{data.asset2} Current Price: {asset2_current:.6f}")
            #print(f"Timestamp:         {asset1_timestamp}")
            #print(f"Price Ratio:       {asset1_current/asset2_current:.6f}")
            
            # Add fill areas for line charts only
            if self.chart_style != 'candlestick':
                ax.fill_between(data.df1.index, asset1_limits[0], data.df1['close'], 
                               alpha=0.1, color=self.theme.primary_color)
                ax_secondary.fill_between(data.df2.index, asset2_limits[0], data.df2['close'], 
                                        alpha=0.1, color=self.theme.secondary_color)
            
            # Add current price annotations
            self._add_price_annotations(ax, ax_secondary, data, statistics)
            
            # Set title
            chart_type = "Candlestick" if self.chart_style == "candlestick" else "Line"
            title = f'{data.asset1} vs {data.asset2} • Dual Y-Axis {chart_type} Chart'
            
            # Apply styling
            self._apply_dual_axis_styling(ax, ax_secondary, data, title)
            
            # Create legend
            self._create_legend(ax, data)
            
            # Format x-axis
            self.format_xaxis_if_last_panel(ax, data, panel_idx, panel_count)
            
            return PanelResult(
                statistics=statistics,
                y_limits=asset1_limits,
                title=title,
                success=True
            )
            
        except Exception as e:
            error_msg = f"Dual axis panel plotting failed: {e}"
            #print(f"[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            return PanelResult(success=False, error_message=error_msg)
    
    def _plot_candlesticks(self, ax: Axes, ax_secondary: Axes, data: PanelData):
        """Plot candlestick charts for both assets"""
        # Calculate candlestick width based on timeframe
        candle_width = self._get_candle_width(data.timeframe)
        
        # Plot candlesticks for asset1 on primary axis
        for i in range(len(data.df1)):
            row = data.df1.iloc[i]
            x = mdates.date2num(data.df1.index[i])
            
            color1 = self.theme.success_color if row['close'] >= row['open'] else self.theme.danger_color
            
            # Draw the high-low line (wick)
            ax.plot([x, x], [row['low'], row['high']], color=color1, linewidth=0.8, alpha=0.9)
            
            # Draw the open-close rectangle (body)
            height = abs(row['close'] - row['open'])
            bottom = min(row['close'], row['open'])
            rect = Rectangle((x - candle_width/2, bottom), candle_width, height, 
                           facecolor=color1, edgecolor=color1, alpha=0.9)
            ax.add_patch(rect)
        
        # Plot candlesticks for asset2 on secondary axis
        for i in range(len(data.df2)):
            row = data.df2.iloc[i]
            x = mdates.date2num(data.df2.index[i])
            
            color2 = self.theme.warning_color if row['close'] >= row['open'] else '#ff6b00'
            
            # Draw the high-low line (wick) on secondary axis
            ax_secondary.plot([x, x], [row['low'], row['high']], color=color2, linewidth=0.8, alpha=0.9)
            
            # Draw the open-close rectangle (body) on secondary axis
            height = abs(row['close'] - row['open'])
            bottom = min(row['close'], row['open'])
            # Note: For secondary axis, we need to create patches differently
            # This is a simplified approach - in practice you might need coordinate transformation
    
    def _plot_lines(self, ax: Axes, ax_secondary: Axes, data: PanelData):
        """Plot line charts for both assets"""
        ax.plot(data.df1.index, data.df1['close'], color=self.theme.primary_color, 
               linewidth=1.5, alpha=0.9, label=f'{data.asset1}')
        ax_secondary.plot(data.df2.index, data.df2['close'], color=self.theme.secondary_color, 
                         linewidth=1.5, alpha=0.9, label=f'{data.asset2}')
    
    def _get_candle_width(self, timeframe: str) -> float:
        """Get appropriate candlestick width based on timeframe"""
        width_map = {
            '1m': 0.0004,
            '5m': 0.002,
            '15m': 0.007,
            '1h': 0.03,
            '4h': 0.125,
            '1d': 0.6
        }
        return width_map.get(timeframe, 0.03)
    
    def _calculate_price_limits(self, df: pd.DataFrame) -> tuple:
        """Calculate appropriate price limits with padding"""
        price_min = df[['open', 'high', 'low', 'close']].min().min()
        price_max = df[['open', 'high', 'low', 'close']].max().max()
        padding = (price_max - price_min) * 0.05
        return (price_min - padding, price_max + padding)
    
    def _add_price_annotations(self, ax: Axes, ax_secondary: Axes, data: PanelData, stats: Dict):
        """Add current price annotations"""
        ax.text(0.02, 0.95, f'{data.asset1}: {stats["asset1_current"]:.6f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=self.theme.bg_color, 
                         alpha=0.9, edgecolor=self.theme.primary_color),
                color=self.theme.primary_color, fontsize=11, fontweight='bold')
        
        ax_secondary.text(0.98, 0.95, f'{data.asset2}: {stats["asset2_current"]:.6f}', 
                         transform=ax_secondary.transAxes, verticalalignment='top', 
                         horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor=self.theme.bg_color, 
                                  alpha=0.9, edgecolor=self.theme.secondary_color),
                         color=self.theme.secondary_color, fontsize=11, fontweight='bold')
    
    def _apply_dual_axis_styling(self, ax: Axes, ax_secondary: Axes, data: PanelData, title: str):
        """Apply styling to dual axis chart"""
        # Main axis styling
        ax.grid(True, alpha=0.3, color=self.theme.grid_color, linestyle='-', linewidth=0.5)
        ax.set_facecolor(self.theme.bg_color)
        ax.spines['top'].set_visible(False)
        ax.set_title(title, color=self.theme.text_color, fontsize=14, 
                    fontweight='normal', pad=15, loc='left')
        
        # Left Y-axis for asset1
        ax.set_ylabel(f'{data.asset1} Price', color=self.theme.primary_color, fontsize=11)
        ax.tick_params(axis='y', labelcolor=self.theme.primary_color, which='both')
        
        # Right Y-axis for asset2
        ax_secondary.set_ylabel(f'{data.asset2} Price', color=self.theme.secondary_color, fontsize=11)
        ax_secondary.tick_params(axis='y', labelcolor=self.theme.secondary_color, which='both')
        ax_secondary.spines['top'].set_visible(False)
        
        # Ensure secondary axis is properly positioned and visible
        ax_secondary.yaxis.set_label_position('right')
        ax_secondary.yaxis.tick_right()
        
        #print(f"[STYLING DEBUG] Applied dual axis styling - {data.asset1} (left), {data.asset2} (right)")
    
    def _create_legend(self, ax: Axes, data: PanelData):
        """Create appropriate legend based on chart style"""
        if self.chart_style == 'candlestick':
            handles = [plt.Line2D([0], [0], color=self.theme.success_color, linewidth=2, label=data.asset1),
                      plt.Line2D([0], [0], color=self.theme.secondary_color, linewidth=2, label=data.asset2)]
            ax.legend(handles=handles, loc='upper left', frameon=False, fontsize=10)
        else:
            ax.legend(loc='upper left', frameon=False, fontsize=10)