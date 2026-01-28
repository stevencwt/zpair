#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chart Orchestrator Module - Reusable Chart Generation Engine
Provides a clean API for generating 1, 2, or 3 panel charts using external panel modules
Can be used by main programs, test scripts, or other applications
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import os

# Import reusable panel modules
from panel_base import PanelData, ThemeConfig, PanelRegistry
from fair_value_panel import FairValuePanel
from ratio_panel import RatioPanel
from zscore_panel import ZScorePanel  
from dual_axis_panel import DualAxisPanel

class ChartOrchestrator:
    """
    Reusable chart generation orchestrator
    Manages panel creation, layout, theming, and output generation
    """
    
    def __init__(self, theme: str = 'dark'):
        """
        Initialize the chart orchestrator
        
        Args:
            theme: Chart theme ('light' or 'dark')
        """
        self.theme_config = self._setup_theme(theme)
        self.panel_registry = PanelRegistry()
        self.panel_results = {}
        self._setup_panel_registry()
        
    def _setup_theme(self, theme: str) -> ThemeConfig:
        """Setup theme configuration"""
        if theme == 'dark':
            plt.rcParams['figure.facecolor'] = '#1e1e1e'
            plt.rcParams['axes.facecolor'] = '#1e1e1e'
            plt.rcParams['axes.edgecolor'] = '#404040'
            plt.rcParams['axes.labelcolor'] = 'white'
            plt.rcParams['xtick.color'] = 'white'
            plt.rcParams['ytick.color'] = 'white'
            plt.rcParams['text.color'] = 'white'
            plt.rcParams['grid.color'] = '#404040'
            
            return ThemeConfig(
                bg_color='#1e1e1e',
                text_color='white',
                grid_color='#404040',
                border_color='#404040',
                primary_color='#2962ff',
                secondary_color='#ff9800',
                success_color='#26a69a',
                warning_color='#ff9800',
                danger_color='#f23645'
            )
        else:
            plt.rcParams['figure.facecolor'] = 'white'
            plt.rcParams['axes.facecolor'] = 'white'
            plt.rcParams['axes.edgecolor'] = '#d1d4dc'
            plt.rcParams['axes.labelcolor'] = '#787b86'
            plt.rcParams['xtick.color'] = '#787b86'
            plt.rcParams['ytick.color'] = '#787b86'
            plt.rcParams['text.color'] = '#131722'
            plt.rcParams['grid.color'] = '#e1e3eb'
            
            return ThemeConfig(
                bg_color='white',
                text_color='#131722',
                grid_color='#e1e3eb',
                border_color='#d1d4dc',
                primary_color='#2962ff',
                secondary_color='#ff9800',
                success_color='#26a69a',
                warning_color='#ff9800',
                danger_color='#f23645'
            )
    
    def _setup_panel_registry(self):
        """Register all available panels using external modules"""
        try:
            self.panel_registry.register_panel('fair_value', FairValuePanel)
            self.panel_registry.register_panel('ratio', RatioPanel)
            self.panel_registry.register_panel('zscore', ZScorePanel)
            self.panel_registry.register_panel('dual_axis', DualAxisPanel)
        except Exception as e:
            raise RuntimeError(f"Failed to register panels: {e}")
    
    def get_available_panels(self) -> List[str]:
        """Get list of available panel types"""
        return self.panel_registry.list_available_panels()
    
    def _calculate_volume_bar_width(self, timeframe: str, df_length: int) -> float:
        """Calculate appropriate bar width for volume/spread bars"""
        if df_length == 0:
            return 0.8
        
        timeframe_widths = {
            '1m': pd.Timedelta(seconds=40).total_seconds() / 86400,
            '5m': pd.Timedelta(minutes=3).total_seconds() / 86400,
            '15m': pd.Timedelta(minutes=10).total_seconds() / 86400,
            '1h': pd.Timedelta(minutes=45).total_seconds() / 86400,
            '4h': pd.Timedelta(hours=3).total_seconds() / 86400,
            '1d': 0.8
        }
        return timeframe_widths.get(timeframe, 0.8)
    
    def _prepare_data(self, df1: pd.DataFrame, df2: Optional[pd.DataFrame], 
                     asset1: str, asset2: str, timeframe: str,
                     zscore_window: int = 40, zscore_thresholds: List[int] = None,
                     extra_data: Dict[str, Any] = None) -> PanelData:
        """Prepare data container for panel plotting"""
        
        if zscore_thresholds is None:
            zscore_thresholds = [1, 2, 3]
        
        if extra_data is None:
            extra_data = {}
        
        # Align dataframes if both provided
        if df2 is not None:
            common_index = df1.index.intersection(df2.index)
            df1 = df1.loc[common_index]
            df2 = df2.loc[common_index]
            
            if len(df1) == 0:
                raise ValueError("No common timestamps between assets")
        
        # Calculate volume bar width
        volume_bar_width = self._calculate_volume_bar_width(timeframe, len(df1))
        
        return PanelData(
            df1=df1,
            df2=df2,
            asset1=asset1,
            asset2=asset2,
            timeframe=timeframe,
            zscore_window=zscore_window,
            zscore_thresholds=zscore_thresholds,
            volume_bar_width=volume_bar_width,
            extra_data=extra_data
        )
    
    def create_chart(self, 
                    df1: pd.DataFrame, 
                    df2: Optional[pd.DataFrame] = None,
                    asset1: str = "Asset1", 
                    asset2: str = "Asset2",
                    timeframe: str = "1h",
                    panels: List[str] = None,
                    config: Dict[str, Any] = None,
                    output_path: Optional[str] = None,
                    display: bool = True,
                    figure_size: Tuple[int, int] = None,
                    **kwargs) -> Dict[str, Any]:
        """
        Create a multi-panel chart using specified panels
        (Silenced Version - Debug prints removed)
        """
        
        # Default configuration
        if config is None:
            config = self._get_default_config()
        
        # Default panels
        if panels is None:
            if df2 is not None:
                panels = ['fair_value', 'zscore', 'dual_axis']  # Pair analysis
            else:
                panels = ['dual_axis']  # Single asset
        
        # Validate panels
        available_panels = self.get_available_panels()
        invalid_panels = [p for p in panels if p not in available_panels]
        if invalid_panels:
            raise ValueError(f"Invalid panel types: {invalid_panels}. Available: {available_panels}")
        
        # Check if pair analysis panels are used with single asset
        pair_panels = ['fair_value', 'ratio', 'zscore']
        if df2 is None and any(p in pair_panels for p in panels):
            raise ValueError(f"Panels {pair_panels} require two assets (df2 must be provided)")
        
        # Prepare data
        panel_data = self._prepare_data(
            df1=df1, df2=df2, asset1=asset1, asset2=asset2, 
            timeframe=timeframe,
            zscore_window=kwargs.get('zscore_window', 40),
            zscore_thresholds=kwargs.get('zscore_thresholds', [1, 2, 3]),
            extra_data=kwargs.get('extra_data', {})
        )
        
        # Create figure
        panel_count = len(panels)
        if figure_size is None:
            figure_size = (16, 5 * panel_count + 2)
        
        fig = plt.figure(figsize=figure_size, facecolor=self.theme_config.bg_color)
        gs = fig.add_gridspec(panel_count, 1, hspace=0.15, left=0.08, right=0.92, top=0.95, bottom=0.05)
        
        # Plot each panel
        self.panel_results = {}
        successful_panels = []
        
        for panel_idx, panel_type in enumerate(panels):
            try:
                # Get panel configuration
                panel_config = self._get_panel_config(config, panel_type, kwargs)
                
                # Get panel instance from registry
                panel = self.panel_registry.get_panel(panel_type, panel_config, self.theme_config)
                
                # Create axes
                ax = fig.add_subplot(gs[panel_idx])
                if panel_idx > 0:
                    ax.sharex(fig.axes[0])
                
                # Create secondary axis for panels that need it
                ax_secondary = None
                if panel_type in ['fair_value', 'ratio']:
                    ax_secondary = ax.twinx()
                
                # Plot panel
                result = panel.plot(ax, ax_secondary, panel_data, panel_idx, panel_count)
                
                # Store results
                self.panel_results[panel_type] = result
                
                if result.success:
                    successful_panels.append(panel_type)
                
            except Exception as e:
                # Critical errors should still be reported for system health
                print(f"[ERROR] Failed to plot panel {panel_type}: {e}")
        
        # Generate results summary
        results = {
            'success': len(successful_panels) == len(panels),
            'successful_panels': successful_panels,
            'failed_panels': [p for p in panels if p not in successful_panels],
            'total_panels': len(panels),
            'panel_results': self.panel_results,
            'data_info': {
                'asset1': asset1,
                'asset2': asset2,
                'timeframe': timeframe,
                'data_points': len(df1),
                'correlation': df1['close'].corr(df2['close']) if df2 is not None else None
            }
        }
        
        # Save or display
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                       facecolor=self.theme_config.bg_color, edgecolor='none')
            results['output_file'] = output_path
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return results
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for panels"""
        return {
            "fair_value": {
                "show_confidence_bands": True,
                "confidence_level": 0.95
            },
            "regression_settings": {
                "window_size": 80,
                "min_periods": 30,
                "rolling_regression": True
            },
            "zscore_settings": {
                "window_size": 40,
                "min_periods": 20,
                "calculation_method": "rolling"
            },
            "chart_style": "line"
        }
    
    def _get_panel_config(self, config: Dict[str, Any], panel_type: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract panel-specific configuration"""
        panel_config = config.copy()
        
        # Add any kwargs as config
        panel_config.update(kwargs)
        
        return panel_config

    def create_single_panel_chart(self, panel_type: str, df1: pd.DataFrame, 
                                 df2: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        """Convenience method for creating single panel charts"""
        return self.create_chart(df1=df1, df2=df2, panels=[panel_type], **kwargs)
    
    def create_dual_panel_chart(self, panel_types: List[str], df1: pd.DataFrame, 
                               df2: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        """Convenience method for creating dual panel charts"""
        if len(panel_types) != 2:
            raise ValueError("Dual panel chart requires exactly 2 panel types")
        return self.create_chart(df1=df1, df2=df2, panels=panel_types, **kwargs)
    
    def create_triple_panel_chart(self, panel_types: List[str], df1: pd.DataFrame, 
                                 df2: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        """Convenience method for creating triple panel charts"""
        if len(panel_types) != 3:
            raise ValueError("Triple panel chart requires exactly 3 panel types")
        return self.create_chart(df1=df1, df2=df2, panels=panel_types, **kwargs)
