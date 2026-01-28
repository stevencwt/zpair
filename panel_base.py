#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Panel Base Classes and Data Transfer Objects for Modular Chart System
Provides abstract base classes and data containers for chart panel system
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

@dataclass
class PanelData:
    """Data container for panel inputs"""
    df1: pd.DataFrame
    df2: Optional[pd.DataFrame] = None
    asset1: str = ""
    asset2: str = ""
    timeframe: str = "1h"
    zscore_window: int = 60
    zscore_thresholds: List[int] = field(default_factory=lambda: [1, 2, 3])
    volume_bar_width: float = 0.8
    extra_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PanelResult:
    """Result container for panel outputs"""
    statistics: Dict[str, Any] = field(default_factory=dict)
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    legend_entries: List[Tuple[str, str]] = field(default_factory=list)
    y_limits: Optional[Tuple[float, float]] = None
    title: str = ""
    success: bool = True
    error_message: str = ""

@dataclass
class ThemeConfig:
    """Theme configuration for consistent styling"""
    bg_color: str = 'white'
    text_color: str = '#131722'
    grid_color: str = '#e1e3eb'
    border_color: str = '#d1d4dc'
    primary_color: str = '#2962ff'
    secondary_color: str = '#ff9800'
    success_color: str = '#26a69a'
    warning_color: str = '#ff9800'
    danger_color: str = '#f23645'

class PanelBase(ABC):
    """Abstract base class for all chart panels"""
    
    def __init__(self, panel_config: Dict, theme_config: ThemeConfig):
        self.config = panel_config
        self.theme = theme_config
        self.name = self.__class__.__name__
    
    @abstractmethod
    def plot(self, ax: Axes, ax_secondary: Optional[Axes], 
             data: PanelData, panel_idx: int, panel_count: int) -> PanelResult:
        """
        Main plotting method - must be implemented by subclasses
        
        Args:
            ax: Primary matplotlib axis
            ax_secondary: Secondary axis (for dual-axis plots)
            data: Input data container
            panel_idx: Current panel index (0-based)
            panel_count: Total number of panels
            
        Returns:
            PanelResult with statistics and metadata
        """
        pass
    
    @abstractmethod
    def get_required_data_fields(self) -> List[str]:
        """Return list of required fields in input dataframes"""
        pass
    
    def validate_data(self, data: PanelData) -> Tuple[bool, str]:
        """
        Validate input data for this panel
        
        Returns:
            (is_valid, error_message)
        """
        if data.df1 is None or len(data.df1) == 0:
            return False, "Primary dataframe is empty or None"
        
        required_fields = self.get_required_data_fields()
        missing_fields = []
        
        for field in required_fields:
            if field not in data.df1.columns:
                missing_fields.append(field)
        
        if missing_fields:
            return False, f"Missing required fields in df1: {missing_fields}"
        
        return True, ""
    
    def calculate_basic_statistics(self, series: pd.Series, name: str) -> Dict[str, float]:
        """Calculate basic statistics for a data series"""
        if len(series) == 0:
            return {}
        
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return {}
        
        return {
            f"{name}_min": float(clean_series.min()),
            f"{name}_max": float(clean_series.max()),
            f"{name}_mean": float(clean_series.mean()),
            f"{name}_std": float(clean_series.std()),
            f"{name}_latest": float(clean_series.iloc[-1]) if len(clean_series) > 0 else 0.0,
            f"{name}_count": len(clean_series)
        }
    
    def format_xaxis_if_last_panel(self, ax: Axes, data: PanelData, 
                                   panel_idx: int, panel_count: int):
        """Apply x-axis formatting if this is the last panel"""
        if panel_idx == panel_count - 1:
            self._format_xaxis(ax, data.df1, data.timeframe)
        else:
            ax.tick_params(axis='x', labelbottom=False)
    
    def _format_xaxis(self, ax: Axes, df: pd.DataFrame, timeframe: str):
        """Format x-axis based on timeframe - extracted from original code"""
        import matplotlib.dates as mdates
        
        if timeframe == '1d':
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            if len(df) > 365:
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            elif len(df) > 180:
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            else:
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        elif timeframe == '4h':
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            if len(df) > 100:
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            else:
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            if len(df) > 200:
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
            elif len(df) > 100:
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            else:
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=9)
    
    def apply_common_styling(self, ax: Axes, title: str = "", ylabel: str = ""):
        """Apply common styling to axis"""
        ax.grid(True, alpha=0.3, color=self.theme.grid_color, linestyle='-', linewidth=0.5)
        ax.set_facecolor(self.theme.bg_color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if title:
            ax.set_title(title, color=self.theme.text_color, fontsize=14, 
                        fontweight='normal', pad=15, loc='left')
        
        if ylabel:
            ax.set_ylabel(ylabel, color=self.theme.text_color, fontsize=11)
    
    def add_text_annotation(self, ax: Axes, text: str, position: Tuple[float, float] = (0.25, 0.85),
                           edge_color: str = None, text_color: str = None):
        """Add text annotation box to axis"""
        if edge_color is None:
            edge_color = self.theme.primary_color
        if text_color is None:
            text_color = self.theme.text_color
            
        ax.text(position[0], position[1], text, 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=self.theme.bg_color, 
                         alpha=0.9, edgecolor=edge_color),
                color=text_color, fontsize=11, fontweight='bold')

class PanelRegistry:
    """Registry for managing available panel types"""
    
    def __init__(self):
        self._panels: Dict[str, type] = {}
    
    def register_panel(self, name: str, panel_class: type):
        """Register a panel class with a name"""
        if not issubclass(panel_class, PanelBase):
            raise ValueError(f"Panel class {panel_class} must inherit from PanelBase")
        self._panels[name] = panel_class
    
    def get_panel(self, name: str, panel_config: Dict, theme_config: ThemeConfig) -> PanelBase:
        """Get a panel instance by name"""
        if name not in self._panels:
            raise ValueError(f"Panel '{name}' not found. Available: {list(self._panels.keys())}")
        
        panel_class = self._panels[name]
        return panel_class(panel_config, theme_config)
    
    def list_available_panels(self) -> List[str]:
        """List all registered panel names"""
        return list(self._panels.keys())
    
    def is_registered(self, name: str) -> bool:
        """Check if a panel name is registered"""
        return name in self._panels

# Global panel registry instance
panel_registry = PanelRegistry()