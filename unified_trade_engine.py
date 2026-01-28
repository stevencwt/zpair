#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Trade Engine - Single Track Execution
-------------------------------------------
Replaces DualTrackSimulator. Handles both SIM and REAL modes via a single logic path.
UPDATED: Added Pair Trading Support (Legs & Beta Sizing).
PATCHED: Enforced EQUAL VALUE SIZING (Dollar Neutral) as per user request.
INTEGRATED: Direct TradeExecutionEngine (No HTTP).
"""

import json
import time
import os
# import requests # REMOVED: Replaced by Direct Engine Integration
from pathlib import Path
from datetime import datetime, timezone

def utcnow_iso():
    return datetime.now(timezone.utc).isoformat()

class UnifiedTradeEngine:
    def __init__(self, config, config_path='config_SOL_MELANIA_5m1m.json'):
        self.config = config
        self.config_path = config_path  # [PHASE 1] Store for config updates
        self.mode = config.get('trading', {}).get('mode', 'SIM')  # REAL or SIM
        self.asset = config.get('asset', 'SOL/MELANIA')
        # self.listener_url = ... # REMOVED: No longer used
        
        # [PHASE 2] Direct Execution Engine Integration
        self.exec_engine = None
        if self.mode == 'REAL':
            try:
                print("[UNIFIED] 🚀 Initializing REAL TRADING Connection...")
                from hyper_market_adapter import HyperMarketAdapter
                from trade_execution_engine import TradeExecutionEngine
                
                # Load Credentials
                hl_cfg = config.get('trading', {}).get('hyperliquid', {})
                private_key = hl_cfg.get('private_key')
                wallet = hl_cfg.get('wallet_address')
                
                if not private_key or not wallet:
                    raise ValueError("Missing Hyperliquid credentials in config")

                # Initialize Adapter & Engine
                self.market_adapter = HyperMarketAdapter(private_key, wallet)
                self.exec_engine = TradeExecutionEngine(self.market_adapter)
                print("[UNIFIED] ✅ TradeExecutionEngine Attached via HyperMarketAdapter")
                
            except Exception as e:
                print(f"[UNIFIED] ❌ FAILED to initialize Real Trading: {e}")
                print("[UNIFIED] ⚠️  Falling back to SIM mode for safety.")
                self.mode = 'SIM'
        
        # [PHASE 1] Initialize config updater for mode persistence
        try:
            from config_updater import ConfigUpdater
            self.config_updater = ConfigUpdater(config_path)
            print(f"[UNIFIED ENGINE] Config updater initialized")
            print(f"[UNIFIED ENGINE] Backups: {self.config_updater.backup_dir}")
        except Exception as e:
            print(f"[UNIFIED ENGINE] Warning: Config updater not available: {e}")
            self.config_updater = None
        
        # [PHASE 1] Mode switching state
        self.mode_switch_pending = None
        
        # Storage Setup
        storage_cfg = config.get('trade_simulation', {}).get('storage', {})
        
        # 1. Determine Directory
        dir_name = storage_cfg.get('positions_directory')
        if dir_name:
            self.storage_dir = Path(dir_name)
        else:
            self.storage_dir = Path(config.get('snapshot_dir', 'market_data'))
            
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # 2. Determine Filenames based on Mode (SIM vs REAL)
        if self.mode == 'REAL':
            open_name = storage_cfg.get('real_positions_file', 'real_open_positions.json')
            closed_name = storage_cfg.get('real_closed_positions_file', 'real_closed_positions.json')
        else:
            open_name = storage_cfg.get('sim_positions_file', 'sim_open_positions.json')
            closed_name = storage_cfg.get('sim_closed_positions_file', 'sim_closed_positions.json')

        self.open_pos_file = self.storage_dir / open_name
        self.closed_pos_file = self.storage_dir / closed_name
        
        # Load State
        self.open_positions = self._load_json(self.open_pos_file)
        self.closed_positions = self._load_json(self.closed_pos_file)
        
        print(f"[UNIFIED ENGINE] Storage: {self.storage_dir}")
        print(f"[UNIFIED ENGINE] Files: Open={open_name}, Closed={closed_name}")
        print(f"[UNIFIED ENGINE] Initialized in {self.mode} mode. Loaded {len(self.open_positions)} active positions.")
        
        # [PHASE 1] Load pending mode switch state (must be after storage_dir is set)
        self._load_pending_mode_switch()

    def _load_json(self, filepath):
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[UNIFIED] Error loading {filepath}: {e}")
        return []

    def _save_state(self):
        """Persist state to disk"""
        try:
            with open(self.open_pos_file, 'w') as f:
                json.dump(self.open_positions, f, indent=2)
            with open(self.closed_pos_file, 'w') as f:
                json.dump(self.closed_positions, f, indent=2)
        except Exception as e:
            print(f"[UNIFIED] Error saving state: {e}")

    # REMOVED: _send_http_command (Replaced by direct Engine calls)

    def _update_pnl(self, current_base_price, current_quote_price=None):
        """
        Update Floating P&L for all open positions using LEG-BASED calculation.
        Total PnL = PnL(Base Leg) + PnL(Quote Leg)
        """
        if current_base_price <= 0: return

        state_changed = False
        
        for pos in self.open_positions:
            # Check if this position has detailed legs (New Format)
            if 'legs' in pos and isinstance(pos['legs'], list) and len(pos['legs']) == 2:
                # --- PAIRS TRADING P&L ---
                if current_quote_price is None or current_quote_price <= 0:
                    continue 

                total_pnl = 0.0
                
                # Update Base Leg (Index 0)
                leg_base = pos['legs'][0]
                lb_qty = float(leg_base.get('qty', 0))
                lb_entry = float(leg_base.get('entry_price', 0))
                
                if leg_base['direction'] == 'LONG':
                    base_pnl = (current_base_price - lb_entry) * lb_qty
                else:
                    base_pnl = (lb_entry - current_base_price) * lb_qty
                
                leg_base['current_price'] = current_base_price
                leg_base['pnl'] = base_pnl
                total_pnl += base_pnl
                
                # Update Quote Leg (Index 1)
                leg_quote = pos['legs'][1]
                lq_qty = float(leg_quote.get('qty', 0))
                lq_entry = float(leg_quote.get('entry_price', 0))
                
                # Use current quote price if available, else keep old
                curr_q = current_quote_price if current_quote_price > 0 else lq_entry
                
                if leg_quote['direction'] == 'LONG':
                    quote_pnl = (curr_q - lq_entry) * lq_qty
                else:
                    quote_pnl = (lq_entry - curr_q) * lq_qty
                    
                leg_quote['current_price'] = curr_q
                leg_quote['pnl'] = quote_pnl
                total_pnl += quote_pnl
                
                # Update Main Position
                old_pnl = pos.get('pnl', 0.0)
                if abs(total_pnl - old_pnl) > 0.0001:
                    pos['pnl'] = total_pnl
                    pos['current_price'] = current_base_price # Tracking base as primary
                    state_changed = True

            else:
                # --- LEGACY / SINGLE ASSET FALLBACK ---
                try:
                    entry_price = float(pos.get('entry_price', 0.0))
                    size = float(pos.get('size', 1.0))
                    direction = pos.get('direction', 'LONG').upper() 
                except (ValueError, TypeError):
                    continue
                
                if entry_price <= 0: entry_price = current_base_price
                
                if direction == 'SHORT':
                    pnl = (entry_price - current_base_price) * size
                else:
                    pnl = (current_base_price - entry_price) * size
                
                old_pnl = pos.get('pnl', 0.0)
                if abs(pnl - old_pnl) > 0.0001:
                    pos['pnl'] = pnl
                    pos['current_price'] = current_base_price
                    state_changed = True

        if state_changed:
            self._save_state()

    def sync_state(self, marketstate):
        """
        Synchronize Portfolio State.
        """
        # 1. Update P&L with granular prices
        prices = marketstate.get('prices', {})
        base_price = prices.get('primary', 0.0)
        quote_price = prices.get('reference', 0.0)
        
        self._update_pnl(base_price, quote_price)

        # 2. Inject Aggregated Metrics & Direction
        total_count = len(self.open_positions)
        overall_direction = "LONG"
        if self.open_positions:
            overall_direction = self.open_positions[0].get('direction', 'LONG')

        for pos in self.open_positions:
            pos['aggregated_metrics'] = {
                'actual_component_count': total_count,
                'direction': pos.get('direction', overall_direction),
                'total_pnl': sum(p.get('pnl', 0.0) for p in self.open_positions)
            }

        # 3. Inject Simplified State into Marketstate
        marketstate['portfolio'] = {
            'mode': self.mode,
            'count': len(self.open_positions),
            'positions': self.open_positions,
            'total_pnl': sum(p.get('pnl', 0.0) for p in self.open_positions)
        }
        
        marketstate['trade_simulation'] = {
            'open_aggregated_positions': {
                'simulation': {'positions': self.open_positions}, 
                'real': {'positions': self.open_positions}
            }
        }

    def process(self, marketstate):
        """
        Main Loop: Execute Signals.
        """
        # 1. Extract Prices & Metrics
        prices = marketstate.get('prices', {})
        base_price = prices.get('primary', 0.0)
        quote_price = prices.get('reference', 0.0)
        
        # Extract Beta (Try multiple locations)
        beta = 1.0
        if 'regression_metrics' in marketstate:
            beta = marketstate['regression_metrics'].get('beta', 1.0)
        elif 'regression' in marketstate:
            beta = marketstate['regression'].get('beta', 1.0)

        # Update P&L
        self._update_pnl(base_price, quote_price)
        
        # 2. Process Signals
        signal = marketstate.get('trading_signal', {}).get('current_recommendation', {})
        rec = signal.get('recommendation', 'WAIT')
        action = signal.get('action', None)

        # Extract entry z-score for position tracking (needed for scaling strategies)
        entry_zscore = marketstate.get('regression', {}).get('zscore', 0.0)
        
        # Handle ADD_POSITION recommendation (scaling signal from ModularDecisionEngine)
        if rec == "ADD_POSITION":
            print(f"[UNIFIED] ADD_POSITION signal detected - scaling position (current: {len(self.open_positions)})")
            rec = "ENTER"  # Convert to ENTER for _handle_signal processing
            # Ensure direction matches existing positions when scaling
            if 'direction' not in signal and self.open_positions:
                signal['direction'] = self.open_positions[0].get('direction', 'LONG')
                print(f"[UNIFIED] Auto-matched scaling direction to existing position: {signal['direction']}")
        
        # Legacy action-based detection (fallback for old signal format)
        elif action == "ADD_POSITION":
            rec = "ENTER"
            if 'direction' not in signal and self.open_positions:
                signal['direction'] = self.open_positions[0].get('direction', 'LONG')

        if rec in ['ENTER', 'EXIT', 'panic_sell', 'take_profit', 'cut_loss']:
            self._handle_signal(rec, signal, base_price, quote_price, beta, entry_zscore)
            
        # 3. Re-Inject State
        self.sync_state(marketstate)

    def _handle_signal(self, rec, signal_data, base_price, quote_price, beta, entry_zscore=0.0):
        """Execute logic with Leg-Aware Sizing"""
        
        # Logic for ENTRY
        if rec == 'ENTER': 
            if len(self.open_positions) >= 5:
                print(f"[UNIFIED] Max positions reached (5). Ignoring ENTER.")
                return

            direction = signal_data.get('direction', 'LONG')

            # --- CALCULATE LEGS (EQUAL VALUE) ---
            # Get base position size from config (Default $1000)
            sizing_cfg = self.config.get('position_sizing', {}).get('sizing_params', {}).get('htf_beta', {})
            base_usd = float(sizing_cfg.get('base_position_usd', 1000.0))
            
            # [FIX] EQUAL VALUE SIZING (Dollar Neutral)
            # Leg 1: Base Asset
            base_qty = base_usd / base_price if base_price > 0 else 0
            
            # Leg 2: Quote Asset (Allocated SAME USD Value as Base)
            quote_usd = base_usd 
            quote_qty = quote_usd / quote_price if quote_price > 0 else 0

            # Define Legs
            leg_base = {
                "asset": self.asset.split('/')[0],
                "direction": direction,
                "qty": base_qty,
                "entry_price": base_price,
                "value_usd": base_usd,
                "pnl": 0.0
            }
            
            leg_quote = {
                "asset": self.asset.split('/')[1] if '/' in self.asset else "QUOTE",
                "direction": "SHORT" if direction == "LONG" else "LONG", # Opposite
                "qty": quote_qty,
                "entry_price": quote_price,
                "value_usd": quote_usd,
                "pnl": 0.0
            }

            # Create Position Object
            new_pos = {
                'id': f"POS-{int(time.time())}-{len(self.open_positions)}",
                'asset': self.asset,
                'entry_price': base_price, 
                'entry_zscore': entry_zscore,
                'beta_at_entry': beta,
                'size': 1.0, 
                'timestamp': utcnow_iso(),
                'timestamps': {
                    'most_recent_add': utcnow_iso() if self.open_positions else None
                },
                'pnl': 0.0,
                'direction': direction,
                'type': 'SCALING' if self.open_positions else 'INITIAL',
                'legs': [leg_base, leg_quote]
            }
            
            if self.mode == 'REAL' and self.exec_engine:
                # [PHASE 2] EXECUTE VIA ENGINE DIRECTLY
                assets = self.asset.split('/')
                
                # Determine Directions
                if direction == 'LONG':
                    l1_dir, l2_dir = 'long', 'short'
                else:
                    l1_dir, l2_dir = 'short', 'long'

                # Execute
                result = self.exec_engine.execute_atomic_pair_by_value(
                    l1_dir, assets[0], base_usd,
                    l2_dir, assets[1], quote_usd,
                    leverage=self.config.get('trading', {}).get('leverage', 5)
                )

                if not result['success']:
                    print(f"[UNIFIED] ❌ Execution Failed: {result['message']}")
                    return # Stop: Do not record position
            
            self.open_positions.append(new_pos)
            self._save_state()
            print(f"[UNIFIED] Opened Pair {new_pos['id']} ({direction}) | Base: {base_qty:.4f} (${base_usd:.0f}) | Quote: {quote_qty:.4f} (${quote_usd:.0f}) | Beta: {beta:.2f}")

        # Logic for EXIT
        elif rec in ['EXIT', 'panic_sell', 'take_profit', 'cut_loss'] and self.open_positions:
            if self.mode == 'REAL' and self.exec_engine:
                # [PHASE 2] EXECUTE VIA ENGINE DIRECTLY (CLOSE ALL)
                assets = self.asset.split('/')
                # Leverage passed for potential rollback re-opening
                leverage = self.config.get('trading', {}).get('leverage', 5)
                
                result = self.exec_engine.execute_atomic_close_all(
                    assets[0], assets[1], leverage
                )
                
                if not result['success']:
                    print(f"[UNIFIED] ❌ Close Failed: {result['message']}")
                    return # Stop: Do not remove position

            for pos in self.open_positions:
                pos['close_price'] = base_price
                pos['closed_at'] = utcnow_iso()
                pos['exit_reason'] = rec
                
                # Snapshot closing prices for legs
                if 'legs' in pos:
                    pos['legs'][0]['exit_price'] = base_price
                    pos['legs'][1]['exit_price'] = quote_price
                
                self.closed_positions.append(pos)
            
            self.open_positions = []
            self._save_state()
            print(f"[UNIFIED] Closed All Positions @ {base_price} / {quote_price}")

    # ========================================================================
    # PHASE 1: MODE SWITCHING SYSTEM
    # ========================================================================
    
    def _persist_mode_to_config(self, new_mode):
        """
        Phase 1: Persist mode change to config file
        Uses config_updater.py (Phase 0) for safe updates
        
        Args:
            new_mode: 'SIM' or 'REAL'
            
        Returns:
            True if successful
        """
        if not self.config_updater:
            print(f"[MODE_SWITCH] ⚠️  Config updater not available")
            return False
        
        try:
            success = self.config_updater.update('trading.mode', new_mode)
            
            if success:
                print(f"[MODE_SWITCH] ✅ Mode persisted to config file: {new_mode}")
                # Update in-memory config
                self.config['trading']['mode'] = new_mode
                return True
            else:
                print(f"[MODE_SWITCH] ❌ Failed to persist mode to config")
                return False
                
        except Exception as e:
            print(f"[MODE_SWITCH] ❌ Error persisting mode: {e}")
            return False
    
    def _update_file_paths(self):
        """Update position file paths based on current mode"""
        storage_cfg = self.config.get('trade_simulation', {}).get('storage', {})
        
        if self.mode == 'REAL':
            open_name = storage_cfg.get('real_positions_file', 'real_open_positions.json')
            closed_name = storage_cfg.get('real_closed_positions_file', 'real_closed_positions.json')
        else:
            open_name = storage_cfg.get('sim_positions_file', 'sim_open_positions.json')
            closed_name = storage_cfg.get('sim_closed_positions_file', 'sim_closed_positions.json')
        
        self.open_pos_file = self.storage_dir / open_name
        self.closed_pos_file = self.storage_dir / closed_name
        
        print(f"[MODE_SWITCH] Updated file paths: Open={open_name}, Closed={closed_name}")
    
    def switch_mode_hard(self, target_mode, confirmation_text=None):
        """
        CASE 1 & 3: Hard mode switch - Force close all positions immediately
        
        Args:
            target_mode: 'SIM' or 'REAL'
            confirmation_text: Required for REAL→SIM switches
            
        Returns:
            Dict with status and details
        """
        # Validation
        if target_mode not in ['SIM', 'REAL']:
            return {
                'status': 'ERROR',
                'message': f'Invalid mode: {target_mode}. Must be SIM or REAL'
            }
        
        if target_mode == self.mode:
            return {
                'status': 'NO_CHANGE',
                'message': f'Already in {target_mode} mode'
            }
        
        # CRITICAL SAFETY: Block REAL→SIM if positions exist
        if self.mode == 'REAL' and target_mode == 'SIM':
            if self.open_positions:
                position_summary = [
                    {
                        'id': p['id'],
                        'pnl': p.get('pnl', 0.0),
                        'direction': p.get('direction', 'UNKNOWN')
                    }
                    for p in self.open_positions
                ]
                
                return {
                    'status': 'ERROR',
                    'message': 'Cannot switch from REAL to SIM while positions are open',
                    'instruction': 'Please close all positions first using CLOSE_ALL_POSITIONS, then switch to SIM mode',
                    'open_positions_count': len(self.open_positions),
                    'positions': position_summary,
                    'suggested_steps': [
                        '1. Send command: {"action": "CLOSE_ALL_POSITIONS"}',
                        '2. Wait for all positions to close',
                        '3. Send command: {"action": "SWITCH_MODE", "mode": "SIM", "method": "HARD"}'
                    ]
                }
        # Force close all positions
        positions_closed = 0
        if self.open_positions:
            print(f"[MODE_SWITCH] Hard switch: Closing {len(self.open_positions)} {self.mode} position(s)")
            
            # Get current price for close
            current_price = self.open_positions[0].get('current_price', self.open_positions[0].get('entry_price', 0.0))
            
            for pos in self.open_positions:
                pos['close_price'] = current_price
                pos['closed_at'] = utcnow_iso()
                pos['exit_reason'] = f'MODE_SWITCH_HARD_{self.mode}_TO_{target_mode}'
                self.closed_positions.append(pos)
                positions_closed += 1
            
            self.open_positions = []
            self._save_state()
            
            print(f"[MODE_SWITCH] ✅ Closed {positions_closed} position(s)")
        
        # Switch mode
        old_mode = self.mode
        self.mode = target_mode
        
        # Update file paths
        self._update_file_paths()
        
        # Reload positions from new mode's files
        self.open_positions = self._load_json(self.open_pos_file)
        
        # Persist to config
        config_updated = self._persist_mode_to_config(target_mode)
        
        print(f"[MODE_SWITCH] ✅ Hard switch complete: {old_mode} → {target_mode}")
        
        # Log to audit file
        self._log_mode_switch('HARD', old_mode, target_mode, positions_closed)
        
        return {
            'status': 'SUCCESS',
            'message': f'Switched from {old_mode} to {target_mode}',
            'method': 'HARD',
            'positions_closed': positions_closed,
            'config_updated': config_updated,
            'current_mode': self.mode,
            'timestamp': utcnow_iso()
        }
    
    def switch_mode_soft(self, target_mode, timeout_minutes=60):
        """
        CASE 2 & 4: Soft mode switch - Wait for positions to close naturally
        
        Args:
            target_mode: 'SIM' or 'REAL'
            timeout_minutes: How long to wait before fallback to hard switch
            
        Returns:
            Dict with status
        """
        # Validation
        if target_mode not in ['SIM', 'REAL']:
            return {
                'status': 'ERROR',
                'message': f'Invalid mode: {target_mode}. Must be SIM or REAL'
            }
        
        if target_mode == self.mode:
            return {
                'status': 'NO_CHANGE',
                'message': f'Already in {target_mode} mode'
            }
        
        # If no positions, switch immediately
        if not self.open_positions:
            print(f"[MODE_SWITCH] No open positions - switching immediately")
            return self.switch_mode_hard(target_mode)
        
        # Enter waiting state
        from datetime import timedelta
        self.mode_switch_pending = {
            'target_mode': target_mode,
            'method': 'SOFT',
            'initiated_at': utcnow_iso(),
            'timeout_at': (datetime.now(timezone.utc) + timedelta(minutes=timeout_minutes)).isoformat(),
            'timeout_minutes': timeout_minutes,
            'positions_waiting': [p['id'] for p in self.open_positions],
            'position_count_at_start': len(self.open_positions)
        }
        
        # Save pending state (survives restart)
        self._save_mode_switch_state()
        
        print(f"[MODE_SWITCH] Soft switch initiated: {self.mode} → {target_mode}")
        print(f"[MODE_SWITCH] Waiting for {len(self.open_positions)} position(s) to close naturally")
        print(f"[MODE_SWITCH] Timeout: {timeout_minutes} minutes")
        
        return {
            'status': 'PENDING',
            'message': f'Soft switch pending - waiting for positions to close',
            'target_mode': target_mode,
            'current_mode': self.mode,
            'positions_waiting': len(self.open_positions),
            'timeout_minutes': timeout_minutes,
            'timeout_at': self.mode_switch_pending['timeout_at']
        }
    
    def check_soft_switch_completion(self):
        """
        Monitor soft switch progress - MUST be called in main loop every cycle
        
        Completes switch when:
        1. All positions closed naturally, OR
        2. Timeout reached (fallback to hard switch)
        """
        if not self.mode_switch_pending:
            return
        
        target_mode = self.mode_switch_pending['target_mode']
        
        # Check if all positions closed
        if not self.open_positions:
            print(f"[MODE_SWITCH] ✅ All positions closed naturally")
            print(f"[MODE_SWITCH] Completing soft switch: {self.mode} → {target_mode}")
            
            # Complete the switch
            old_mode = self.mode
            self.mode = target_mode
            
            # Update file paths and reload
            self._update_file_paths()
            self.open_positions = self._load_json(self.open_pos_file)
            
            # Persist to config
            config_updated = self._persist_mode_to_config(target_mode)
            
            # Clear pending state
            self.mode_switch_pending = None
            self._save_mode_switch_state()
            
            print(f"[MODE_SWITCH] ✅ Soft switch complete: {old_mode} → {target_mode}")
            
            # Log to audit file
            self._log_mode_switch('SOFT', old_mode, target_mode, 0)
            
            return
        
        # Check timeout
        now = datetime.now(timezone.utc)
        timeout_at = datetime.fromisoformat(self.mode_switch_pending['timeout_at'])
        
        if now >= timeout_at:
            print(f"[MODE_SWITCH] ⚠️  Soft switch timeout reached")
            print(f"[MODE_SWITCH] {len(self.open_positions)} position(s) still open")
            print(f"[MODE_SWITCH] Falling back to HARD switch")
            
            # Clear pending state
            self.mode_switch_pending = None
            self._save_mode_switch_state()
            
            # Fallback to hard switch
            self.switch_mode_hard(target_mode)
    
    def _save_mode_switch_state(self):
        """Save pending mode switch state to disk (survives restart)"""
        state_file = self.storage_dir / 'mode_switch_state.json'
        
        state = {
            'current_mode': self.mode,
            'pending_switch': self.mode_switch_pending,
            'last_updated': utcnow_iso()
        }
        
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"[MODE_SWITCH] Warning: Could not save state: {e}")
    
    def _load_pending_mode_switch(self):
        """Load pending mode switch state on startup"""
        state_file = self.storage_dir / 'mode_switch_state.json'
        
        if not state_file.exists():
            return
        
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            pending = state.get('pending_switch')
            if pending:
                self.mode_switch_pending = pending
                print(f"[MODE_SWITCH] Resumed pending soft switch from previous session")
                print(f"[MODE_SWITCH] Target: {pending['target_mode']}, Initiated: {pending['initiated_at']}")
        except Exception as e:
            print(f"[MODE_SWITCH] Warning: Could not load pending state: {e}")
    
    def _log_mode_switch(self, method, old_mode, new_mode, positions_closed):
        """Log mode switch to audit file"""
        audit_file = self.storage_dir / 'mode_switch_audit.log'
        
        try:
            with open(audit_file, 'a') as f:
                f.write(f"{utcnow_iso()} | {method} | {old_mode}→{new_mode} | Closed: {positions_closed}\n")
        except Exception as e:
            print(f"[MODE_SWITCH] Warning: Could not log audit: {e}")
    
    # ========================================================================
    # END OF PHASE 1 MODE SWITCHING
    # ========================================================================

    def check_target_profit_reached(self, position):
        """
        Check if a position has reached the configured target profit.
        
        Args:
            position: Position dict with 'pnl' and 'legs' fields
            
        Returns:
            bool: True if target is reached, False otherwise
        """
        target_config = self.config.get('trading', {}).get('target_profit', {})
        
        if not target_config.get('enabled', False):
            return False
        
        profit_type = target_config.get('type', 'PERCENTAGE')
        target_value = float(target_config.get('value', 0.0))
        
        current_pnl = position.get('pnl', 0.0)
        
        if profit_type == 'PERCENTAGE':
            # Calculate percentage profit based on entry value
            legs = position.get('legs', [])
            if legs:
                entry_value = legs[0].get('value_usd', 0.0)
                if entry_value > 0:
                    profit_pct = (current_pnl / entry_value) * 100
                    return profit_pct >= target_value
        else:  # USD
            return current_pnl >= target_value
        
        return False

    def manual_override(self, action, data):
        """Handle Manual Commands"""
        
        # [PHASE 1] Mode switching commands
        if action == "SWITCH_MODE":
            target_mode = data.get('mode', 'SIM').upper()
            method = data.get('method', 'HARD').upper()
            confirmation = data.get('confirmation', '')
            timeout_minutes = data.get('timeout_minutes', 60)
            
            if method == 'HARD':
                result = self.switch_mode_hard(target_mode, confirmation)
            elif method == 'SOFT':
                result = self.switch_mode_soft(target_mode, timeout_minutes)
            else:
                result = {
                    'status': 'ERROR',
                    'message': f'Invalid method: {method}. Must be HARD or SOFT'
                }
            
            return result
        
        if action == "PANIC_SELL":
            price = float(data.get('price', 0.0)) # This assumes base price
            timestamp = utcnow_iso()
            for pos in self.open_positions:
                pos['close_price'] = price
                pos['closed_at'] = timestamp
                pos['exit_reason'] = 'MANUAL_PANIC_SELL'
                self.closed_positions.append(pos)
            
            self.open_positions = []
            self._save_state()
            print("[UNIFIED] ⚠️ PANIC SELL EXECUTED")
            return True

        elif action == "SYNC_POSITION":
            # Simple sync clearing
            if self.open_positions:
                self.open_positions = []
                self._save_state()
            return True

        elif action == "IMPORT_POSITION":
            # Basic import (Simplified)
            new_pos = {
                'id': f"IMP-{int(time.time())}",
                'asset': self.asset,
                'entry_price': float(data.get('real_entry_px', 0.0)),
                'size': float(data.get('real_qty', 0.0)),
                'timestamp': utcnow_iso(),
                'pnl': 0.0,
                'direction': 'LONG' if float(data.get('real_qty', 0)) > 0 else 'SHORT',
                'strategy': 'manual_import'
            }
            self.open_positions.append(new_pos)
            self._save_state()
            print(f"[UNIFIED] Imported Position: {new_pos['id']}")
            return True
        
        elif action == "CLOSE_ALL_POSITIONS":
            """
            Close all open positions using atomic execution engine.
            - SIM mode: Closes positions in local storage
            - REAL mode: Executes via TradeExecutionEngine with rollback protection
            """
            print(f"[UNIFIED] CLOSE_ALL_POSITIONS command received (Mode: {self.mode})")
            
            if not self.open_positions:
                print(f"[UNIFIED] No open positions to close")
                return {
                    'status': 'SUCCESS',
                    'message': 'No open positions',
                    'positions_closed': 0
                }
            
            positions_closed = 0
            timestamp = utcnow_iso()
            
            # REAL MODE: Use atomic execution engine
            if self.mode == 'REAL' and self.exec_engine:
                try:
                    # Parse asset pair (e.g., "SOL/MELANIA" -> ["SOL", "MELANIA"])
                    assets = self.asset.split('/')
                    if len(assets) < 2:
                        # Fallback for single asset
                        leg1_asset, leg2_asset = self.asset, None
                    else:
                        leg1_asset, leg2_asset = assets[0].strip(), assets[1].strip()
                    
                    # Get leverage from config
                    leverage = int(self.config.get('trading', {}).get('leverage', 1))
                    
                    print(f"[UNIFIED] Executing atomic close: {leg1_asset}/{leg2_asset} at {leverage}x leverage")
                    
                    # Execute atomic close through the execution engine
                    result = self.exec_engine.execute_atomic_close_all(
                        leg1_asset=leg1_asset,
                        leg2_asset=leg2_asset if leg2_asset else "USD", 
                        leverage=leverage
                    )
                    
                    if result.get('success'):
                        print(f"[UNIFIED] ✅ Atomic close successful")
                        
                        # Update local position tracking to reflect close
                        current_price = 0.0 
                        for pos in self.open_positions:
                            pos['close_price'] = current_price
                            pos['closed_at'] = timestamp
                            pos['exit_reason'] = 'MANUAL_CLOSE_ALL_REAL'
                            self.closed_positions.append(pos)
                            positions_closed += 1
                        
                        self.open_positions = []
                        self._save_state()
                        
                        return {
                            'status': 'SUCCESS',
                            'message': 'All REAL positions closed atomically',
                            'positions_closed': positions_closed,
                            'execution_data': result.get('data', {})
                        }
                    else:
                        print(f"[UNIFIED] ❌ Atomic close failed: {result.get('message')}")
                        return {
                            'status': 'ERROR',
                            'message': f"Atomic close failed: {result.get('message')}",
                            'execution_data': result.get('state', {})
                        }
                        
                except Exception as e:
                    print(f"[UNIFIED] ❌ Error executing REAL close: {e}")
                    import traceback
                    traceback.print_exc()
                    return {
                        'status': 'ERROR',
                        'message': f"Execution error: {str(e)}"
                    }
            
            # SIM MODE: Close positions in local storage only
            else:
                print(f"[UNIFIED] Closing {len(self.open_positions)} SIM position(s)")
                
                for pos in self.open_positions:
                    pos['close_price'] = pos.get('current_price', pos.get('entry_price', 0.0))
                    pos['closed_at'] = timestamp
                    pos['exit_reason'] = 'MANUAL_CLOSE_ALL_SIM'
                    self.closed_positions.append(pos)
                    positions_closed += 1
                
                self.open_positions = []
                self._save_state()
                
                print(f"[UNIFIED] ✅ Closed {positions_closed} SIM position(s)")
                
                return {
                    'status': 'SUCCESS',
                    'message': f'Closed {positions_closed} SIM positions',
                    'positions_closed': positions_closed
                }
        
        elif action == "SET_TARGET_PROFIT":
            """
            Set target profit threshold.
            CORRECTED: Maps to 'take_profit_strategies.{active_strat}.dollar_target'
            to match config_SOL_MELANIA_5m1m.json structure.
            """
            print(f"[UNIFIED] SET_TARGET_PROFIT command received")
            
            try:
                # 1. Parse Input (Dashboard sends 'dollar_target')
                target = float(data.get('dollar_target', 0.0))
                
                if target <= 0:
                    return {'status': 'ERROR', 'message': 'Target profit value must be positive'}

                # 2. Identify Active Strategy (e.g., 'hit_and_run_profit')
                active_strat = self.config.get('active_strategies', {}).get('take_profit', 'hit_and_run_profit')
                
                # 3. Update In-Memory Config (Instant Dashboard Reflection)
                if 'take_profit_strategies' not in self.config:
                    self.config['take_profit_strategies'] = {}
                if active_strat not in self.config['take_profit_strategies']:
                    self.config['take_profit_strategies'][active_strat] = {}
                
                self.config['take_profit_strategies'][active_strat]['dollar_target'] = target
                
                # 4. Persist to Disk (Preserve strict structure)
                if self.config_updater:
                    # Construct specific dot-notation path
                    path = f"take_profit_strategies.{active_strat}.dollar_target"
                    success = self.config_updater.update(path, target)
                    
                    if success:
                        print(f"[UNIFIED] ✅ Profit Target Updated: ${target:.2f} (Strategy: {active_strat})")
                        return {
                            'status': 'SUCCESS',
                            'message': f'Target profit updated to ${target:.2f}',
                            'strategy': active_strat
                        }
                    else:
                        return {'status': 'ERROR', 'message': 'Failed to persist configuration'}
                else:
                    return {'status': 'ERROR', 'message': 'Config updater not available'}
                    
            except Exception as e:
                print(f"[UNIFIED] ❌ Error setting target profit: {e}")
                return {'status': 'ERROR', 'message': f'Configuration error: {str(e)}'}
            
        return False