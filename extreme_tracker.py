#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extreme Tracker - Multi-Timeframe Z-Score Extreme History Management
UPDATED: Tracks top 8 extreme values (4 positive + 4 negative) per timeframe
Position scaling uses configured primary timeframe only
Raw extreme values for primary timeframe copied to marketstate JSON
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from collections import deque


class ExtremeTracker:
    """
    Tracks top 8 z-score extremes (4 positive + 4 negative) across multiple timeframes 
    with persistent JSON storage. Pre-populates with synthetic extreme values when no 
    historical data exists. Position scaling uses configured primary timeframe only.
    """
    
    def __init__(self, strategy_config: Dict[str, Any], snapshot_dir: str, primary_timeframe: str):
        """
        Initialize multi-timeframe extreme tracker.
        
        Args:
            strategy_config: Strategy-specific configuration section
            snapshot_dir: Directory where marketstate files are stored
            primary_timeframe: Primary timeframe for position scaling (e.g., "1m", "5s", "1h")
        """
        # Extract scaling config section
        scaling_config = strategy_config.get('position_scaling_config', {})
        extreme_config = scaling_config.get('extreme_tracking', {})
        
        self.window_size = 8  # Fixed at 8 extremes per timeframe
        self.min_threshold = extreme_config.get('min_threshold', 1.0)
        self.extreme_history_depth = extreme_config.get('history_depth', 3)
        self.max_age_hours = extreme_config.get('max_age_hours', 24)
        self.min_extremes = extreme_config.get('min_extremes_required', 2)
        self.retention_hours = extreme_config.get('retention_hours', 168)  # 7 days
        
        # Storage configuration - unified location with marketstate
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.storage_path = self.snapshot_dir / 'extremes_history.json'
        
        # Primary timeframe for position scaling
        self.primary_timeframe = primary_timeframe
        
        # Supported timeframes
        self.supported_timeframes = ['5s', '1m', '1h']
        
        # Multi-timeframe storage: dict of deques (fixed at 8 elements each)
        self.extremes: Dict[str, deque] = {}
        for tf in self.supported_timeframes:
            self.extremes[tf] = deque(maxlen=self.window_size)
        
        print(f"[EXTREME_TRACKER] Multi-timeframe tracker initialized")
        print(f"  Storage: {self.storage_path}")
        print(f"  Primary timeframe (for scaling): {self.primary_timeframe}")
        print(f"  Supported timeframes: {self.supported_timeframes}")
        print(f"  Max extremes per timeframe: {self.window_size}")
        print(f"  History depth: {self.extreme_history_depth}")
        print(f"  Min threshold: {self.min_threshold}")
        print(f"  Max age: {self.max_age_hours}h")
        
        # Try to load existing state
        loaded = self.load()
        
        # Check if we need to pre-populate with synthetic data
        if not loaded or self._needs_synthetic_data():
            self._pre_populate_synthetic_extremes()
            self.persist()
            print(f"[EXTREME_TRACKER] Pre-populated with synthetic extreme values")
            
            # Show statistics for all timeframes
            for tf in self.supported_timeframes:
                stats = self.get_statistics(tf)
                print(f"  {tf}: {stats['total_extremes']} extremes "
                      f"(+{stats['positive_extremes']}/-{stats['negative_extremes']}, "
                      f"{stats['synthetic_extremes']} synthetic)")
    
    def _needs_synthetic_data(self) -> bool:
        """Check if any timeframe needs synthetic data based on current extremes."""
        for tf in self.supported_timeframes:
            fresh_extremes = self._get_fresh_extremes(tf)
            
            # Count positive and negative extremes
            positive_count = sum(1 for e in fresh_extremes if e['zscore'] > self.min_threshold)
            negative_count = sum(1 for e in fresh_extremes if e['zscore'] < -self.min_threshold)
            
            # Need at least min_extremes in each direction for any timeframe
            if positive_count < self.min_extremes or negative_count < self.min_extremes:
                return True
        
        return False
    
    def _pre_populate_synthetic_extremes(self):
        """Pre-populate all timeframes with synthetic extreme values - top 8 per timeframe."""
        
        current_time = datetime.now(timezone.utc)
        
        # Define synthetic extreme values for different timeframes (4 positive + 4 negative each)
        synthetic_extremes_by_tf = {
            '5s': [  # More volatile, wider extremes
                {'zscore': -0.35, 'label': 'Very strong LONG extreme (5s)'},
                {'zscore': -0.36, 'label': 'Strong LONG extreme (5s)'},
                {'zscore': -0.37, 'label': 'Moderate LONG extreme (5s)'},
                {'zscore': -0.38, 'label': 'Conservative LONG extreme (5s)'},
                {'zscore': 0.35, 'label': 'Conservative SHORT extreme (5s)'},
                {'zscore': 0.36, 'label': 'Moderate SHORT extreme (5s)'},
                {'zscore': 0.37, 'label': 'Strong SHORT extreme (5s)'},
                {'zscore': 0.38, 'label': 'Very strong SHORT extreme (5s)'},
            ],
            '1m': [  # Standard extremes
                {'zscore': -0.35, 'label': 'Very strong LONG extreme (1m)'},
                {'zscore': -0.36, 'label': 'Strong LONG extreme (1m)'},
                {'zscore': -0.37, 'label': 'Moderate LONG extreme (1m)'},
                {'zscore': -0.38, 'label': 'Conservative LONG extreme (1m)'},
                {'zscore': 0.35, 'label': 'Conservative SHORT extreme (1m)'},
                {'zscore': 0.36, 'label': 'Moderate SHORT extreme (1m)'},
                {'zscore': 0.37, 'label': 'Strong SHORT extreme (1m)'},
                {'zscore': 0.38, 'label': 'Very strong SHORT extreme (1m)'},
            ],
            '1h': [  # More conservative, tighter extremes
                {'zscore': -0.35, 'label': 'Very strong LONG extreme (1h)'},
                {'zscore': -0.36, 'label': 'Strong LONG extreme (1h)'},
                {'zscore': -0.37, 'label': 'Moderate LONG extreme (1h)'},
                {'zscore': -0.38, 'label': 'Conservative LONG extreme (1h)'},
                {'zscore': 0.35, 'label': 'Conservative SHORT extreme (1h)'},
                {'zscore': 0.36, 'label': 'Moderate SHORT extreme (1h)'},
                {'zscore': 0.37, 'label': 'Strong SHORT extreme (1h)'},
                {'zscore': 0.38, 'label': 'Very strong SHORT extreme (1h)'},
            ]
        }
        
        # Clear existing extremes to start fresh
        for tf in self.supported_timeframes:
            self.extremes[tf].clear()
        
        # Add synthetic extremes with staggered timestamps for each timeframe
        for tf in self.supported_timeframes:
            extremes_data = synthetic_extremes_by_tf[tf]
            
            for i, extreme_data in enumerate(extremes_data):
                # Stagger timestamps by 2 hours each to simulate historical collection
                hours_ago = (len(extremes_data) - i) * 2
                timestamp = current_time - timedelta(hours=hours_ago)
                
                extreme_record = {
                    'zscore': float(extreme_data['zscore']),
                    'timestamp': timestamp.isoformat(),
                    'timeframe': tf,
                    'synthetic': True,
                    'label': extreme_data['label']
                }
                self.extremes[tf].append(extreme_record)
        
        total_synthetic = sum(len(synthetic_extremes_by_tf[tf]) for tf in self.supported_timeframes)
        print(f"[EXTREME_TRACKER] Added {total_synthetic} synthetic extreme values across all timeframes")
    
    def update(self, timeframe: str, zscore: float, timestamp: str) -> None:
        """
        Add z-score and maintain top 8 extremes (4 positive + 4 negative) for the timeframe.
        Tracks extremes by magnitude while preserving positive/negative balance.
        
        Args:
            timeframe: Timeframe identifier ('5s', '1m', '1h')
            zscore: Z-score value to potentially track
            timestamp: ISO timestamp string
        """
        if timeframe not in self.supported_timeframes:
            print(f"[EXTREME_TRACKER WARNING] Unsupported timeframe: {timeframe}")
            return
        
        # Only track if meets minimum threshold
        if abs(zscore) >= self.min_threshold:
            extreme_record = {
                'zscore': float(zscore),
                'timestamp': timestamp,
                'timeframe': timeframe,
                'synthetic': False  # Mark as real data
            }
            
            # Add to deque
            self.extremes[timeframe].append(extreme_record)
            
            # Process to maintain top 8 (4 positive + 4 negative)
            current_list = list(self.extremes[timeframe])
            
            # Separate positive and negative
            positives = [x for x in current_list if x['zscore'] > 0]
            negatives = [x for x in current_list if x['zscore'] < 0]
            
            # Keep top 4 of each by magnitude
            positives.sort(key=lambda x: x['zscore'], reverse=True)  # Highest positive first
            negatives.sort(key=lambda x: x['zscore'])  # Most negative first
            
            top_positives = positives[:4]
            top_negatives = negatives[:4]
            
            # Clear and repopulate deque with top 8 extremes
            self.extremes[timeframe].clear()
            for record in top_positives + top_negatives:
                self.extremes[timeframe].append(record)
    
    def get_nth_least_extreme(self, n: int, direction: str, timeframe: Optional[str] = None) -> Optional[float]:
        """
        Get Nth least extreme z-score for progressive thresholds from specified timeframe.
        
        Args:
            n: Rank (1 = least extreme, N = most extreme in top N)
            direction: 'LONG' or 'SHORT'
            timeframe: Specific timeframe to use (defaults to primary_timeframe)
            
        Returns:
            Z-score threshold or None if insufficient extremes
        """
        if timeframe is None:
            timeframe = self.primary_timeframe
        
        if timeframe not in self.supported_timeframes:
            print(f"[EXTREME_TRACKER ERROR] Invalid timeframe: {timeframe}")
            return None
        
        if n < 1 or n > self.extreme_history_depth:
            return None
        
        # Get fresh extremes only (includes synthetic if within time window)
        fresh_extremes = self._get_fresh_extremes(timeframe)
        
        if direction == 'LONG':
            # For LONG, we care about negative extremes (buying opportunities)
            relevant = [e['zscore'] for e in fresh_extremes if e['zscore'] < -self.min_threshold]
            if len(relevant) < n:
                # Fallback to default if insufficient data
                return -1.5 * n  # Progressive defaults: -1.5, -3.0, -4.5
            # Sort from most negative to least negative
            relevant.sort()
            # Get nth from the end (least extreme)
            return relevant[-n]
        else:  # SHORT
            # For SHORT, we care about positive extremes (shorting opportunities)
            relevant = [e['zscore'] for e in fresh_extremes if e['zscore'] > self.min_threshold]
            if len(relevant) < n:
                # Fallback to default if insufficient data
                return 1.5 * n  # Progressive defaults: 1.5, 3.0, 4.5
            # Sort from most positive to least positive
            relevant.sort(reverse=True)
            # Get nth from the end (least extreme)
            return relevant[-n]
    
    def has_sufficient_fresh_data(
        self,
        min_extremes: int,
        max_age_hours: int,
        direction: str,
        timeframe: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Validate sufficient recent extremes exist for scaling decisions.
        
        Args:
            min_extremes: Minimum number of extremes required
            max_age_hours: Maximum age in hours
            direction: 'LONG' or 'SHORT'
            timeframe: Specific timeframe to check (defaults to primary_timeframe)
            
        Returns:
            (is_sufficient, reason)
        """
        if timeframe is None:
            timeframe = self.primary_timeframe
        
        if timeframe not in self.supported_timeframes:
            return False, f"Invalid timeframe: {timeframe}"
        
        fresh_extremes = self._get_fresh_extremes(timeframe, max_age_hours)
        
        if direction == 'LONG':
            relevant = [e for e in fresh_extremes if e['zscore'] < -self.min_threshold]
        else:
            relevant = [e for e in fresh_extremes if e['zscore'] > self.min_threshold]
        
        count = len(relevant)
        
        # Count synthetic vs real
        synthetic_count = sum(1 for e in relevant if e.get('synthetic', False))
        real_count = count - synthetic_count
        
        if count < min_extremes:
            return False, f"Insufficient fresh extremes for {timeframe}: {count} < {min_extremes}"
        
        data_type = "synthetic" if synthetic_count > real_count else "real"
        return True, f"Sufficient data for {timeframe}: {count} extremes ({real_count} real, {synthetic_count} synthetic) within {max_age_hours}h"
    
    def is_stabilized(
        self,
        stabilization_minutes: int,
        last_add_timestamp: Optional[str]
    ) -> bool:
        """
        Check if enough time passed since last add-on.
        
        Args:
            stabilization_minutes: Minimum minutes required between add-ons
            last_add_timestamp: ISO timestamp of most recent add-on (None if first add)
            
        Returns:
            True if current time - last_add_timestamp >= stabilization_minutes
        """
        if last_add_timestamp is None:
            return True  # First add-on, no stabilization needed
        
        try:
            last_add_dt = datetime.fromisoformat(last_add_timestamp.replace('Z', '+00:00'))
            current_dt = datetime.now(timezone.utc)
            
            elapsed_minutes = (current_dt - last_add_dt).total_seconds() / 60
            
            return elapsed_minutes >= stabilization_minutes
        except:
            return True  # If parsing fails, allow the add-on
    
    def get_marketstate_summary(self) -> Dict[str, Any]:
        """
        Get summary of extremes for all timeframes for inclusion in marketstate JSON.
        Returns top 3 most recent extremes per timeframe for dashboard visibility.
        
        Returns:
            Dictionary with recent extremes for each timeframe
        """
        summary = {}
        
        for tf in self.supported_timeframes:
            fresh_extremes = self._get_fresh_extremes(tf)
            
            # Get recent extremes (sorted by timestamp, most recent first)
            extremes_by_time = sorted(fresh_extremes, key=lambda x: x['timestamp'], reverse=True)
            recent_extremes = [e['zscore'] for e in extremes_by_time[:3]]
            
            summary[tf] = {
                'recent_extremes': recent_extremes,
                'total_count': len(fresh_extremes),
                'synthetic_count': sum(1 for e in fresh_extremes if e.get('synthetic', False)),
                'real_count': sum(1 for e in fresh_extremes if not e.get('synthetic', False))
            }
        
        return summary
    
    def get_primary_timeframe_raw_extremes(self) -> List[Dict[str, Any]]:
        """
        Get all raw extreme values for the primary timeframe only.
        Returns all tracked extremes (up to 8) for the primary timeframe.
        
        Returns:
            List of all extreme records for primary timeframe
        """
        if self.primary_timeframe not in self.supported_timeframes:
            return []
        
        # Return all extreme records for primary timeframe
        return list(self.extremes[self.primary_timeframe])
    
    def persist(self) -> None:
        """
        Save multi-timeframe extremes to JSON with format:
        {
          "timeframes": {
            "5s": {"extremes": [...], "metadata": {...}},
            "1m": {"extremes": [...], "metadata": {...}},
            "1h": {"extremes": [...], "metadata": {...}}
          },
          "global_metadata": {...}
        }
        """
        current_time = datetime.now(timezone.utc).isoformat()
        
        state = {
            'timeframes': {},
            'global_metadata': {
                'last_updated': current_time,
                'primary_timeframe': self.primary_timeframe,
                'supported_timeframes': self.supported_timeframes,
                'retention_hours': self.retention_hours,
                'window_size': self.window_size,
                'min_threshold': self.min_threshold,
                'extreme_history_depth': self.extreme_history_depth,
                'version': '3.1',  # Updated version for top-8 tracking
                'storage_location': 'unified_with_marketstate'
            }
        }
        
        # Save data for each timeframe
        for tf in self.supported_timeframes:
            extremes_list = list(self.extremes[tf])
            has_synthetic = any(e.get('synthetic', False) for e in extremes_list)
            
            state['timeframes'][tf] = {
                'extremes': extremes_list,
                'metadata': {
                    'last_updated': current_time,
                    'timeframe': tf,
                    'total_extremes': len(extremes_list),
                    'has_synthetic': has_synthetic,
                    'is_primary_timeframe': (tf == self.primary_timeframe)
                }
            }
        
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(state, f, indent=2)
            # print(f"[EXTREME_TRACKER] Persisted multi-timeframe extremes to {self.storage_path}")
        except Exception as e:
            print(f"[EXTREME_TRACKER ERROR] Failed to persist state: {e}")
    
    def load(self) -> bool:
        """
        Load multi-timeframe extremes from JSON on startup.
        Handles both old single-timeframe and new multi-timeframe formats.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.storage_path.exists():
            print(f"[EXTREME_TRACKER] No existing state found, starting fresh")
            return False
        
        try:
            with open(self.storage_path, 'r') as f:
                state = json.load(f)
            
            # Check if this is multi-timeframe format
            if 'timeframes' in state and 'global_metadata' in state:
                # Multi-timeframe format - load all timeframes and apply top-8 filtering
                for tf in self.supported_timeframes:
                    tf_data = state['timeframes'].get(tf, {})
                    extremes_list = tf_data.get('extremes', [])
                    
                    # Apply top-8 filtering during load to handle legacy data
                    filtered_extremes = self._filter_to_top_8_extremes(extremes_list)
                    
                    self.extremes[tf] = deque(filtered_extremes, maxlen=self.window_size)
                
                print(f"[EXTREME_TRACKER] Loaded multi-timeframe extremes from storage")
                for tf in self.supported_timeframes:
                    count = len(self.extremes[tf])
                    if count > 0:
                        synthetic_count = sum(1 for e in self.extremes[tf] if e.get('synthetic', False))
                        print(f"  {tf}: {count} extremes ({count - synthetic_count} real, {synthetic_count} synthetic)")
                
            elif 'extremes' in state and 'metadata' in state:
                # Old single-timeframe format - migrate to primary timeframe
                print(f"[EXTREME_TRACKER] Migrating old single-timeframe data to {self.primary_timeframe}")
                
                extremes_list = state['extremes']
                
                # Add timeframe info to old records if missing
                for extreme in extremes_list:
                    if 'timeframe' not in extreme:
                        extreme['timeframe'] = self.primary_timeframe
                
                # Apply top-8 filtering and store in primary timeframe
                filtered_extremes = self._filter_to_top_8_extremes(extremes_list)
                self.extremes[self.primary_timeframe] = deque(filtered_extremes, maxlen=self.window_size)
                
                print(f"  Migrated {len(filtered_extremes)} extremes to {self.primary_timeframe}")
                
            else:
                print(f"[EXTREME_TRACKER WARNING] Invalid state format, rebuilding")
                return False
            
            # Cleanup old extremes for all timeframes
            for tf in self.supported_timeframes:
                self.cleanup_old_extremes(tf, self.retention_hours)
            
            return True
            
        except Exception as e:
            print(f"[EXTREME_TRACKER ERROR] Failed to load state: {e}")
            return False
    
    def _filter_to_top_8_extremes(self, extremes_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter extremes list to top 8 (4 positive + 4 negative).
        Used during loading to handle legacy data with more than 8 extremes.
        """
        if len(extremes_list) <= 8:
            return extremes_list
        
        # Separate positive and negative
        positives = [x for x in extremes_list if x['zscore'] > 0]
        negatives = [x for x in extremes_list if x['zscore'] < 0]
        
        # Sort and take top 4 of each
        positives.sort(key=lambda x: x['zscore'], reverse=True)  # Highest positive first
        negatives.sort(key=lambda x: x['zscore'])  # Most negative first
        
        top_positives = positives[:4]
        top_negatives = negatives[:4]
        
        return top_positives + top_negatives
    
    def cleanup_old_extremes(self, timeframe: str, max_age_hours: int) -> None:
        """
        Remove extremes older than retention period for specific timeframe.
        
        Args:
            timeframe: Timeframe to clean up
            max_age_hours: Maximum age in hours
        """
        if timeframe not in self.supported_timeframes:
            return
        
        if not self.extremes[timeframe]:
            return
        
        current_time = datetime.now(timezone.utc)
        cutoff_time = current_time.timestamp() - (max_age_hours * 3600)
        
        # Filter extremes
        cleaned_extremes = []
        for extreme in self.extremes[timeframe]:
            try:
                extreme_time = datetime.fromisoformat(extreme['timestamp'].replace('Z', '+00:00'))
                # Keep if within time window OR if synthetic and less than 7 days old
                if extreme_time.timestamp() >= cutoff_time:
                    cleaned_extremes.append(extreme)
                elif extreme.get('synthetic', False) and (current_time - extreme_time).days < 7:
                    cleaned_extremes.append(extreme)
            except:
                # Keep if timestamp parsing fails
                cleaned_extremes.append(extreme)
        
        removed_count = len(self.extremes[timeframe]) - len(cleaned_extremes)
        if removed_count > 0:
            self.extremes[timeframe] = deque(cleaned_extremes, maxlen=self.window_size)
            print(f"[EXTREME_TRACKER] Cleaned {removed_count} old extremes from {timeframe}")
    
    def _get_fresh_extremes(self, timeframe: str, max_age_hours: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get extremes within freshness window for specific timeframe.
        
        Args:
            timeframe: Timeframe to check
            max_age_hours: Maximum age (uses self.max_age_hours if None)
            
        Returns:
            List of fresh extreme records
        """
        if timeframe not in self.supported_timeframes:
            return []
        
        if max_age_hours is None:
            max_age_hours = self.max_age_hours
        
        current_time = datetime.now(timezone.utc)
        cutoff_time = current_time.timestamp() - (max_age_hours * 3600)
        
        fresh = []
        for extreme in self.extremes[timeframe]:
            try:
                extreme_time = datetime.fromisoformat(extreme['timestamp'].replace('Z', '+00:00'))
                if extreme_time.timestamp() >= cutoff_time:
                    fresh.append(extreme)
            except:
                # Include if timestamp parsing fails
                fresh.append(extreme)
        
        return fresh
    
    def get_statistics(self, timeframe: Optional[str] = None) -> Dict[str, Any]:
        """
        Get extreme tracking statistics for specific timeframe.
        
        Args:
            timeframe: Specific timeframe (defaults to primary_timeframe)
            
        Returns:
            Dictionary with tracking stats for the timeframe
        """
        if timeframe is None:
            timeframe = self.primary_timeframe
        
        if timeframe not in self.supported_timeframes:
            return {
                'error': f'Invalid timeframe: {timeframe}',
                'supported_timeframes': self.supported_timeframes
            }
        
        if not self.extremes[timeframe]:
            return {
                'timeframe': timeframe,
                'total_extremes': 0,
                'positive_extremes': 0,
                'negative_extremes': 0,
                'fresh_extremes': 0,
                'synthetic_extremes': 0,
                'real_extremes': 0
            }
        
        extremes_list = list(self.extremes[timeframe])
        fresh_extremes = self._get_fresh_extremes(timeframe)
        
        positive = [e['zscore'] for e in extremes_list if e['zscore'] > 0]
        negative = [e['zscore'] for e in extremes_list if e['zscore'] < 0]
        synthetic_count = sum(1 for e in extremes_list if e.get('synthetic', False))
        real_count = len(extremes_list) - synthetic_count
        
        return {
            'timeframe': timeframe,
            'total_extremes': len(extremes_list),
            'positive_extremes': len(positive),
            'negative_extremes': len(negative),
            'fresh_extremes': len(fresh_extremes),
            'synthetic_extremes': synthetic_count,
            'real_extremes': real_count,
            'max_positive': max(positive) if positive else 0.0,
            'max_negative': min(negative) if negative else 0.0,
            'oldest_timestamp': extremes_list[0]['timestamp'] if extremes_list else None,
            'newest_timestamp': extremes_list[-1]['timestamp'] if extremes_list else None,
            'is_primary_timeframe': (timeframe == self.primary_timeframe)
        }
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for all timeframes.
        
        Returns:
            Dictionary with stats for each timeframe plus summary
        """
        all_stats = {}
        
        for tf in self.supported_timeframes:
            all_stats[tf] = self.get_statistics(tf)
        
        # Add summary
        total_extremes = sum(stats['total_extremes'] for stats in all_stats.values())
        total_synthetic = sum(stats['synthetic_extremes'] for stats in all_stats.values())
        
        all_stats['summary'] = {
            'primary_timeframe': self.primary_timeframe,
            'total_extremes_all_timeframes': total_extremes,
            'total_synthetic_all_timeframes': total_synthetic,
            'timeframes_with_data': [tf for tf in self.supported_timeframes if all_stats[tf]['total_extremes'] > 0]
        }
        
        return all_stats
