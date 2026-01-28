#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
market_state_manager.py - Persistence Layer Strategy Pattern
============================================================
Abstracts market state I/O to support Hybrid Persistence:
1. Fast Brain: Shared Memory (shm) for <0.1ms IPC latency.
2. Slow Brain: Throttled Disk I/O for safety and dashboard compatibility.
"""

import os
import time
import json
import tempfile
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum
from datetime import date, datetime
import numpy as np
import pandas as pd
from multiprocessing import shared_memory

from debug_utils import DebugLogger, PerformanceTimer, safe_mkdir

# --- 1. JSON Encoder (Robust Version) ---
class CustomJSONEncoder(json.JSONEncoder):
    """Handles Numpy types, Datetime, and Infinity for JSON serialization"""
    def default(self, obj):
        # --- numpy numeric types ---
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            float_val = float(obj)
            # Handle infinity and NaN values
            if not math.isfinite(float_val):
                return 0.0
            return float_val
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # --- boolean types ---
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        # --- datetime / date / pandas timestamp ---
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif pd is not None and isinstance(obj, getattr(pd, "Timestamp", ())):
            return obj.to_pydatetime().isoformat()
        # --- numpy datetime64 ---
        elif isinstance(obj, np.datetime64):
            if pd is not None:
                return pd.to_datetime(obj).to_pydatetime().isoformat()
            return str(obj.astype("datetime64[ns]"))
        # --- enum types ---
        elif isinstance(obj, Enum):
            return obj.value
        # --- set types ---
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        # --- Handle Python float infinity/NaN ---
        elif isinstance(obj, float):
            if not math.isfinite(obj):
                return 0.0
            return obj
            
        return super().default(obj)

    def encode(self, obj):
        """Override encode to catch any remaining infinity values"""
        try:
            return super().encode(obj)
        except ValueError as e:
            if "Out of range float values are not JSON compliant" in str(e):
                # Recursively clean the object if standard encoding fails
                cleaned_obj = self._clean_infinity_values(obj)
                return super().encode(cleaned_obj)
            raise

    def _clean_infinity_values(self, obj):
        """Recursively clean infinity values from nested objects"""
        if isinstance(obj, dict):
            return {k: self._clean_infinity_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_infinity_values(item) for item in obj]
        elif isinstance(obj, float) and not math.isfinite(obj):
            return 0.0
        elif isinstance(obj, np.floating) and not math.isfinite(float(obj)):
            return 0.0
        else:
            return obj

# --- 2. Abstract Strategy Interface ---
class StorageStrategy(ABC):
    @abstractmethod
    def save_snapshot(self, data: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def load_snapshot(self) -> Optional[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def save_barometer(self, data: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        pass

# --- 3. Concrete Strategy: File Storage (Slow Brain) ---
class FileStorageStrategy(StorageStrategy):
    def __init__(self, config: Dict[str, Any], logger: DebugLogger):
        self.logger = logger
        
        # Priority Logic for Directory Path
        default_dir = "marketstate"
        legacy_path = config.get("snapshot_dir")
        new_path_cfg = config.get("market_state_storage", {}).get("file_settings", {}).get("snapshot_dir")
        
        if legacy_path and legacy_path != default_dir:
            path_str = legacy_path
            self.logger.debug_print(f"Using Legacy Config Path: {path_str}", "PERSISTENCE")
        elif new_path_cfg:
            path_str = new_path_cfg
        else:
            path_str = default_dir
             
        self.snapshot_dir = Path(path_str)
        safe_mkdir(str(self.snapshot_dir), self.logger)
        
        self.snapshot_path = self.snapshot_dir / "latest_market_state.json"
        self.barometer_path = self.snapshot_dir / "latest_barometer.json"
        
        self.logger.debug_print(f"File Strategy Active: Writing to {self.snapshot_dir}", "PERSISTENCE")

    def _atomic_write(self, path: Path, record: Dict[str, Any]) -> None:
        try:
            json_str = json.dumps(record, separators=(",", ":"), ensure_ascii=False, cls=CustomJSONEncoder)
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", text=True)
            with os.fdopen(fd, "w", encoding="utf-8") as tmpf:
                tmpf.write(json_str)
            os.replace(tmp, path)
        except Exception as e:
            self.logger.debug_print(f"Atomic write failed: {e}", "PERSISTENCE")

    def save_snapshot(self, data: Dict[str, Any]) -> None:
        self._atomic_write(self.snapshot_path, data)

    def load_snapshot(self) -> Optional[Dict[str, Any]]:
        if self.snapshot_path.exists():
            try:
                with open(self.snapshot_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.logger.debug_print("Loaded snapshot from Disk", "I/O_FILE")
                    return data
            except Exception as e:
                self.logger.debug_print(f"Load failed: {e}", "PERSISTENCE")
        return None

    def save_barometer(self, data: Dict[str, Any]) -> None:
        self._atomic_write(self.barometer_path, data)
        
    def cleanup(self) -> None:
        # Nothing to clean up for files
        pass

# --- 4. Concrete Strategy: Shared Memory (Fast Brain) ---
class MemoryStorageStrategy(StorageStrategy):
    """
    Implements IPC using Shared Memory.
    Layout: [4 bytes Length Header] + [JSON Bytes]
    """
    SHM_NAME = "market_state_shared_fast"
    SHM_SIZE = 4 * 1024 * 1024  # 4MB Capacity
    
    def __init__(self, logger: DebugLogger):
        self.logger = logger
        self.shm = None
        self._initialize_shm()
        self._barometer_block = {} # Keep barometer local for now, or add 2nd SHM if needed

    def _initialize_shm(self):
        """Creates or attaches to shared memory, handling stale blocks."""
        try:
            # 1. Attempt to unlink existing stale memory from crashes
            try:
                existing_shm = shared_memory.SharedMemory(name=self.SHM_NAME, create=False)
                existing_shm.close()
                existing_shm.unlink()
                self.logger.debug_print("Cleaned up stale Shared Memory block", "PERSISTENCE")
            except FileNotFoundError:
                pass # Clean slate

            # 2. Create new memory block
            self.shm = shared_memory.SharedMemory(name=self.SHM_NAME, create=True, size=self.SHM_SIZE)
            self.logger.debug_print(f"Shared Memory Active: '{self.SHM_NAME}' ({self.SHM_SIZE/1024/1024:.0f}MB)", "PERSISTENCE")
            
        except Exception as e:
            self.logger.debug_print(f"FATAL: Failed to initialize Shared Memory: {e}", "PERSISTENCE")
            self.shm = None

    def save_snapshot(self, data: Dict[str, Any]) -> None:
        if not self.shm: return
        
        try:
            # 1. Serialize
            json_str = json.dumps(data, separators=(",", ":"), ensure_ascii=False, cls=CustomJSONEncoder)
            data_bytes = json_str.encode('utf-8')
            data_len = len(data_bytes)

            # 2. Bounds Check
            # We need 4 bytes for header + data length
            if data_len + 4 > self.SHM_SIZE:
                self.logger.debug_print(f"Warning: Data exceeds SHM size ({data_len} > {self.SHM_SIZE})", "PERSISTENCE")
                return

            # 3. Write Header (4 bytes, Big Endian)
            self.shm.buf[:4] = data_len.to_bytes(4, 'big')
            
            # 4. Write Payload
            self.shm.buf[4:4+data_len] = data_bytes
            
        except Exception as e:
            self.logger.debug_print(f"SHM Write Error: {e}", "PERSISTENCE")

    def load_snapshot(self) -> Optional[Dict[str, Any]]:
        """Reads from SHM (used during fast restart)"""
        if not self.shm: return None
        
        try:
            # 1. Read Header
            data_len = int.from_bytes(self.shm.buf[:4], 'big')
            
            if data_len == 0 or data_len + 4 > self.SHM_SIZE:
                return None # Empty or invalid
                
            # 2. Read Payload
            # We create a bytes copy to decode
            data_bytes = bytes(self.shm.buf[4:4+data_len])
            json_str = data_bytes.decode('utf-8')
            
            self.logger.debug_print("Loaded snapshot from Shared Memory", "I/O_MEM")
            return json.loads(json_str)
            
        except Exception as e:
            self.logger.debug_print(f"SHM Read Error: {e}", "PERSISTENCE")
            return None

    def save_barometer(self, data: Dict[str, Any]) -> None:
        self._barometer_block = data.copy()

    def cleanup(self) -> None:
        if self.shm:
            try:
                self.shm.close()
                self.shm.unlink()
                self.logger.debug_print("Shared Memory Unlinked/Freed", "PERSISTENCE")
            except Exception as e:
                self.logger.debug_print(f"Error freeing Shared Memory: {e}", "PERSISTENCE")

# --- 5. The Hybrid Manager (Dual Brain Dispatcher) ---
class MarketStateManager:
    def __init__(self, config: Dict[str, Any], logger: DebugLogger):
        self.config = config
        self.logger = logger
        
        # Initialize BOTH backends
        self.fast_store = MemoryStorageStrategy(logger)
        self.slow_store = FileStorageStrategy(config, logger)
        
        # Throttling Logic for Slow Store
        storage_cfg = config.get("market_state_storage", {}).get("file_settings", {})
        self.write_interval = storage_cfg.get("write_interval_seconds", 5)
        self.last_disk_write_time = 0.0
        
        self.logger.debug_print(f"Hybrid Persistence Initialized. Disk Sync: {self.write_interval}s", "PERSISTENCE")

    def save_snapshot(self, data: Dict[str, Any]) -> None:
        # 1. FAST PATH: Write to Shared Memory (Immediate)
        self.fast_store.save_snapshot(data)
        
        # 2. SLOW PATH: Conditional write to disk (Safety/Dashboard)
        current_time = time.time()
        if current_time - self.last_disk_write_time >= self.write_interval:
            self.slow_store.save_snapshot(data)
            self.last_disk_write_time = current_time

    def load_snapshot(self) -> Optional[Dict[str, Any]]:
        # 1. Try Memory first (Fast Restart)
        data = self.fast_store.load_snapshot()
        
        # 2. Fallback to Disk (Recovery)
        if not data:
            data = self.slow_store.load_snapshot()
            
        return data

    def save_barometer(self, data: Dict[str, Any]) -> None:
        self.fast_store.save_barometer(data)
        self.slow_store.save_barometer(data)

    def shutdown(self):
        """Cleanup resources on exit"""
        self.logger.debug_print("Shutting down persistence layer...", "PERSISTENCE")
        self.fast_store.cleanup()
        self.slow_store.cleanup()