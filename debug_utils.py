#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_utils.py - Debug and Logging Utilities
============================================
Centralized debug logging and performance monitoring for trading systems.
"""

import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

# Define Singapore Time (UTC+8)
SGT = timezone(timedelta(hours=8))

class DebugLogger:
    """Enhanced debug logging with component tracking and performance monitoring"""
    


    def __init__(self, enabled: bool = True, component_prefix: str = "MAIN"):
        self.enabled = enabled
        self.default_component = component_prefix
        self.session_start = time.time()
        self.call_counts = {}
        self.timing_data = {}
    
    def debug_print(self, msg: str, component: Optional[str] = None) -> None:
        """Enhanced debug printing with component tracking"""
        if not self.enabled:
            return
        
        component = component or self.default_component
        timestamp = get_local_time().strftime('%H:%M:%S')
        
        # Track call counts per component
        self.call_counts[component] = self.call_counts.get(component, 0) + 1
        
        print(f"[{component} {timestamp}] {msg}")
        sys.stdout.flush()
    
    def start_timer(self, operation: str, component: Optional[str] = None) -> None:
        """Start timing an operation"""
        component = component or self.default_component
        key = f"{component}:{operation}"
        self.timing_data[key] = {'start': time.time(), 'end': None}
    
    def end_timer(self, operation: str, component: Optional[str] = None) -> float:
        """End timing and return duration"""
        component = component or self.default_component
        key = f"{component}:{operation}"
        
        if key in self.timing_data and self.timing_data[key]['end'] is None:
            end_time = time.time()
            self.timing_data[key]['end'] = end_time
            duration = end_time - self.timing_data[key]['start']
            
            if self.enabled:
                self.debug_print(f"{operation} completed in {duration:.3f}s", component)
            
            return duration
        
        return 0.0
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        session_duration = time.time() - self.session_start
        
        # Calculate timing statistics
        completed_operations = {
            k: v['end'] - v['start'] 
            for k, v in self.timing_data.items() 
            if v['end'] is not None
        }
        
        return {
            "session_duration_seconds": session_duration,
            "debug_calls_by_component": self.call_counts.copy(),
            "operation_timings": completed_operations,
            "total_operations": len(completed_operations)
        }
    
    def log_session_summary(self) -> None:
        """Log session summary statistics"""
        if not self.enabled:
            return
        
        stats = self.get_session_stats()
        self.debug_print("=== SESSION SUMMARY ===")
        self.debug_print(f"Session duration: {stats['session_duration_seconds']:.1f}s")
        self.debug_print(f"Components active: {len(stats['debug_calls_by_component'])}")
        self.debug_print(f"Operations timed: {stats['total_operations']}")
        
        # Show top components by debug call count
        sorted_components = sorted(
            stats['debug_calls_by_component'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for component, count in sorted_components[:5]:  # Top 5
            self.debug_print(f"  {component}: {count} debug calls")


class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, operation_name: str, logger: DebugLogger, component: Optional[str] = None):
        self.operation_name = operation_name
        self.logger = logger
        self.component = component
        self.duration = 0.0
    
    def __enter__(self):
        self.logger.start_timer(self.operation_name, self.component)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = self.logger.end_timer(self.operation_name, self.component)


def get_local_time():
    """Centralized function to get Singapore Local Time."""
    return datetime.now(SGT)

def utcnow_iso() -> str:
    """Update to return SGT ISO for local consistency."""
    return get_local_time().replace(microsecond=0).isoformat()



def safe_mkdir(path: str, logger: Optional[DebugLogger] = None) -> bool:
    """Safely create directory with logging"""
    import os
    
    try:
        os.makedirs(path, exist_ok=True)
        if logger:
            logger.debug_print(f"Created directory: {path}", "FILESYSTEM")
        return True
    except Exception as e:
        if logger:
            logger.debug_print(f"Failed to create directory {path}: {e}", "FILESYSTEM")
        return False


# Legacy compatibility function
def debug_print(msg: str) -> None:
    """Legacy debug print function for backward compatibility"""
    timestamp = get_local_time().strftime('%H:%M:%S')
    print(f"[DEBUG {timestamp}] {msg}")
    sys.stdout.flush()
