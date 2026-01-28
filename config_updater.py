#!/usr/bin/env python3
"""
Safe Config Updater - Phase 0
=============================
Updates specific config values while preserving ALL other fields.
Future-proof: handles unknown fields added later.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime, timezone
from collections import OrderedDict


class ConfigUpdater:
    """Update specific config values while preserving everything else"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.backup_dir = self.config_path.parent / 'config_backups'
        self.backup_dir.mkdir(exist_ok=True)
    
    def update(self, field_path: str, new_value) -> bool:
        """
        Update a nested field while preserving all others
        
        Args:
            field_path: Dot notation path like 'trading.mode'
            new_value: Value to set
            
        Returns:
            True if successful
            
        Example:
            updater.update('trading.mode', 'REAL')
        """
        try:
            # 1. Backup first
            backup = self._backup()
            
            # 2. Load (preserving field order)
            config = self._load()
            
            # 3. Update only the target field
            old_value = self._set_nested(config, field_path, new_value)
            
            # 4. Write atomically
            self._save(config)
            
            # 5. Verify
            if not self._verify(field_path, new_value):
                raise Exception("Verification failed")
            
            print(f"✅ Updated {field_path}: {old_value} → {new_value}")
            print(f"📦 Backup: {backup.name}")
            return True
            
        except Exception as e:
            print(f"❌ Update failed: {e}")
            if backup:
                shutil.copy2(backup, self.config_path)
                print(f"↩️  Restored from backup")
            return False
    
    def get(self, field_path: str, default=None):
        """Get current value"""
        config = self._load()
        return self._get_nested(config, field_path, default)
    
    def _load(self):
        """Load config preserving order"""
        with open(self.config_path) as f:
            return json.load(f, object_pairs_hook=OrderedDict)
    
    def _save(self, config):
        """Save atomically"""
        temp = self.config_path.with_suffix('.tmp')
        with open(temp, 'w') as f:
            json.dump(config, f, indent=2)
        temp.replace(self.config_path)  # Atomic on POSIX
    
    def _get_nested(self, config, path, default=None):
        """Navigate nested dict"""
        current = config
        for key in path.split('.'):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    
    def _set_nested(self, config, path, value):
        """Set nested value, return old value"""
        keys = path.split('.')
        current = config
        
        # Navigate to parent
        for key in keys[:-1]:
            if key not in current:
                current[key] = OrderedDict()
            current = current[key]
        
        # Update final key
        old = current.get(keys[-1])
        current[keys[-1]] = value
        return old
    
    def _backup(self):
        """Create timestamped backup"""
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        backup = self.backup_dir / f"{self.config_path.stem}_{timestamp}.json"
        shutil.copy2(self.config_path, backup)
        
        # Keep last 10 only
        backups = sorted(self.backup_dir.glob('*.json'), 
                        key=lambda p: p.stat().st_mtime, reverse=True)
        for old in backups[10:]:
            old.unlink()
        
        return backup
    
    def _verify(self, path, expected):
        """Verify write succeeded"""
        actual = self.get(path)
        return actual == expected


# Convenience functions
def update_mode(config_path: str, mode: str) -> bool:
    """Quick mode update"""
    if mode not in ['SIM', 'REAL']:
        print(f"Invalid mode: {mode}")
        return False
    return ConfigUpdater(config_path).update('trading.mode', mode)


if __name__ == '__main__':
    # Quick test
    import sys
    if len(sys.argv) > 1:
        config = sys.argv[1]
        updater = ConfigUpdater(config)
        
        # Show current mode
        mode = updater.get('trading.mode')
        print(f"Current mode: {mode}")
        
        # Toggle for test
        new_mode = 'REAL' if mode == 'SIM' else 'SIM'
        print(f"\nTesting update to {new_mode}...")
        updater.update('trading.mode', new_mode)
        
        # Restore
        print(f"\nRestoring to {mode}...")
        updater.update('trading.mode', mode)
        
        print("\n✅ Phase 0 test complete!")
