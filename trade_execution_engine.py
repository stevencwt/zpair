#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trade Execution Engine (v2.1)
-----------------------------
Fix: Corrected price fetching method to match HyperMarketAdapter interface.
"""

import time
from typing import Dict, Any, Optional

class TradeExecutionEngine:
    def __init__(self, market_adapter):
        self.market_adapter = market_adapter
        self.max_retry_attempts = 5
        self.retry_backoff_base = 1.0  # Seconds

    # ==================================================================
    # SECTION 1: ATOMIC PAIR TRADING (Transaction Integrity)
    # ==================================================================

    def execute_atomic_pair_by_value(self, leg1_dir: str, leg1_asset: str, leg1_val: float, 
                                     leg2_dir: str, leg2_asset: str, leg2_val: float, 
                                     leverage: float) -> Dict[str, Any]:
        """
        Executes a pair trade (Open) with Atomic Transaction Integrity.
        """
        print(f"[EXEC ENGINE] Atomic Open: {leg1_dir} {leg1_asset} ${leg1_val} / {leg2_dir} {leg2_asset} ${leg2_val}")
        
        transaction_state = {
            "leg1": {"executed": False, "order_id": None},
            "leg2": {"executed": False, "order_id": None},
            "rollback": {"attempted": False, "successful": False}
        }

        # --- STEP 1: EXECUTE LEG 1 ---
        leg1_res = self._execute_with_retry(
            asset=leg1_asset, direction=leg1_dir, size=leg1_val, 
            leverage=leverage, is_value=True
        )
        
        if leg1_res['success']:
            transaction_state['leg1']['executed'] = True
            transaction_state['leg1']['order_id'] = leg1_res['order_id']
            print(f"[EXEC ENGINE] ✅ Leg 1 ({leg1_asset}) Filled")
        else:
            print(f"[EXEC ENGINE] ❌ Leg 1 Failed. Aborting Transaction.")
            return {"success": False, "message": f"Leg 1 Failed: {leg1_res['message']}", "state": transaction_state}

        # --- STEP 2: EXECUTE LEG 2 ---
        leg2_res = self._execute_with_retry(
            asset=leg2_asset, direction=leg2_dir, size=leg2_val, 
            leverage=leverage, is_value=True
        )
        
        if leg2_res['success']:
            transaction_state['leg2']['executed'] = True
            transaction_state['leg2']['order_id'] = leg2_res['order_id']
            print(f"[EXEC ENGINE] ✅ Leg 2 ({leg2_asset}) Filled. Atomic Trade Complete.")
            return {
                "success": True, 
                "message": f"Atomic Success: {leg1_asset}/{leg2_asset} Opened", 
                "data": transaction_state
            }
        
        # --- STEP 3: ROLLBACK (If Leg 2 Failed) ---
        print(f"[EXEC ENGINE] ⚠️ Leg 2 Failed! Initiating ROLLBACK of Leg 1 ({leg1_asset})...")
        transaction_state['rollback']['attempted'] = True
        
        rollback_dir = 'short' if leg1_dir.lower() == 'long' else 'long'
        
        rollback_res = self._execute_with_retry(
            asset=leg1_asset, direction=rollback_dir, size=leg1_val, 
            leverage=leverage, is_value=True
        )
        
        if rollback_res['success']:
            transaction_state['rollback']['successful'] = True
            print(f"[EXEC ENGINE] ✅ Rollback Successful. Position Neutralized.")
            return {"success": False, "message": "Transaction Failed (Safe Rollback Performed)", "state": transaction_state}
        else:
            print(f"[EXEC ENGINE] 🚨 CRITICAL: ROLLBACK FAILED. MANUAL INTERVENTION REQUIRED.")
            return {"success": False, "message": "CRITICAL: Transaction Failed & Rollback Failed", "state": transaction_state}

    def execute_atomic_close_by_value(self, leg1_asset: str, leg1_val: float, 
                                      leg2_asset: str, leg2_val: float) -> Dict[str, Any]:
        """
        Executes a pair CLOSE by Dollar Value using SMART CLOSE LOGIC.
        """
        print(f"[EXEC ENGINE] Atomic Close (Smart): {leg1_asset} ${leg1_val} / {leg2_asset} ${leg2_val}")
        
        # --- SNAPSHOT STATE ---
        try:
            positions = self.market_adapter.get_current_positions()
        except Exception as e:
            return {"success": False, "message": f"Failed to fetch positions: {e}"}

        transaction_state = {
            "leg1": {"closed": False, "order_id": None},
            "leg2": {"closed": False, "order_id": None}
        }

        # --- STEP 1: CLOSE LEG 1 (Smart) ---
        leg1_res = self._execute_smart_close(leg1_asset, leg1_val, positions.get(leg1_asset))
        
        if leg1_res['success']:
            transaction_state['leg1']['closed'] = True
            transaction_state['leg1']['order_id'] = leg1_res.get('order_id')
        else:
            return {"success": False, "message": f"Leg 1 Close Failed: {leg1_res['message']}", "state": transaction_state}

        # --- STEP 2: CLOSE LEG 2 (Smart) ---
        leg2_res = self._execute_smart_close(leg2_asset, leg2_val, positions.get(leg2_asset))
        
        if leg2_res['success']:
            transaction_state['leg2']['closed'] = True
            transaction_state['leg2']['order_id'] = leg2_res.get('order_id')
            print(f"[EXEC ENGINE] ✅ Atomic Close Complete.")
            return {
                "success": True, 
                "message": f"Atomic Close Success: {leg1_asset}/{leg2_asset}", 
                "data": transaction_state
            }
        
        # --- PARTIAL CLOSE ERROR ---
        print(f"[EXEC ENGINE] ⚠️ Leg 2 Close Failed! Partial Close State: {leg1_asset} Closed, {leg2_asset} Open.")
        return {
            "success": False, 
            "message": "PARTIAL CLOSE ERROR: Leg 1 closed, Leg 2 failed. Manual Check Required.", 
            "state": transaction_state
        }

    def execute_atomic_close_all(self, leg1_asset: str, leg2_asset: str, leverage: float) -> Dict[str, Any]:
        """
        Closes ALL positions for a specific pair with Rollback capability.
        """
        print(f"[EXEC ENGINE] Atomic Close ALL: {leg1_asset} / {leg2_asset}")
        
        try:
            positions = self.market_adapter.get_current_positions()
        except Exception as e:
            return {"success": False, "message": f"Failed to fetch positions: {e}"}

        # Validate Positions
        leg1_data = positions.get(leg1_asset)
        leg2_data = positions.get(leg2_asset)

        if not leg1_data and not leg2_data:
             return {"success": True, "message": "Both positions already closed", "data": {}}

        # Snapshot for Rollback
        leg1_size = abs(float(leg1_data.get('size', 0))) if leg1_data else 0
        leg1_side = 'long' if leg1_data and float(leg1_data.get('size', 0)) > 0 else 'short'
        
        leg2_size = abs(float(leg2_data.get('size', 0))) if leg2_data else 0

        transaction_state = {
            "leg1": {"closed": False},
            "leg2": {"closed": False},
            "rollback": {"attempted": False, "successful": False}
        }

        # --- STEP 1: CLOSE LEG 1 ---
        if leg1_size > 0:
            leg1_res = self._execute_close_with_retry(leg1_asset, leg1_size, is_value=False)
            if not leg1_res['success']:
                return {"success": False, "message": f"Leg 1 Close Failed: {leg1_res['message']}"}
        
        transaction_state['leg1']['closed'] = True
        print(f"[EXEC ENGINE] ✅ Leg 1 ({leg1_asset}) Closed")

        # --- STEP 2: CLOSE LEG 2 ---
        if leg2_size > 0:
            leg2_res = self._execute_close_with_retry(leg2_asset, leg2_size, is_value=False)
            if leg2_res['success']:
                transaction_state['leg2']['closed'] = True
                print(f"[EXEC ENGINE] ✅ Leg 2 ({leg2_asset}) Closed")
                return {"success": True, "message": "Atomic Close All Success", "data": transaction_state}
        else:
             transaction_state['leg2']['closed'] = True
             return {"success": True, "message": "Atomic Close All Success (Leg 2 was empty)", "data": transaction_state}

        # --- STEP 3: ROLLBACK (Re-Open Leg 1) ---
        print(f"[EXEC ENGINE] ⚠️ Leg 2 Close Failed! Re-opening Leg 1...")
        transaction_state['rollback']['attempted'] = True
        
        rollback_res = self._execute_with_retry(
            asset=leg1_asset, direction=leg1_side, size=leg1_size, leverage=leverage, is_value=False
        )
        
        if rollback_res['success']:
            transaction_state['rollback']['successful'] = True
            return {"success": False, "message": "Transaction Failed (Safe Rollback: Leg 1 Re-opened)", "state": transaction_state}
        else:
            return {"success": False, "message": "CRITICAL: Leg 2 Fail + Rollback Fail. Manual Intervention Required.", "state": transaction_state}

    # ==================================================================
    # SECTION 2: INTERNAL HELPERS (Retry + Smart Logic)
    # ==================================================================

    def _execute_smart_close(self, asset, value, position_data):
        """
        Decides whether to close by Value or Quantity (Full Close) based on threshold.
        """
        # 1. Idempotency Check (Already Closed?)
        if not position_data:
            print(f"[EXEC ENGINE] Smart Close: {asset} position not found. Skipping (Success).")
            return {"success": True, "order_id": "ALREADY_CLOSED", "message": "Position already closed"}
        
        size = abs(float(position_data.get('size', 0)))
        if size == 0:
            print(f"[EXEC ENGINE] Smart Close: {asset} size is 0. Skipping (Success).")
            return {"success": True, "order_id": "ALREADY_CLOSED", "message": "Position size is 0"}

        # 2. Smart Threshold Logic
        try:
            # FIX: Use get_current_prices instead of get_price
            price_map = self.market_adapter.get_current_prices([asset])
            price = float(price_map.get(asset, 0))
            
            # Fallback if price fetch failed
            if price == 0: 
                price = float(position_data.get('entry_px', 0))
            
            total_pos_value = size * price
            
            # If requested value is >= 95% of total holding, close FULL SIZE
            if total_pos_value > 0 and (value / total_pos_value) >= 0.95:
                print(f"[EXEC ENGINE] Smart Close {asset}: Request ${value} vs Pos ${total_pos_value:.2f} (>95%) -> CLOSING FULL SIZE {size}")
                return self._execute_close_with_retry(asset, size, is_value=False)
            
            # Otherwise close specific value
            return self._execute_close_with_retry(asset, value, is_value=True)
            
        except Exception as e:
            print(f"[EXEC ENGINE] Smart Close Error: {e}. Defaulting to Value Close.")
            return self._execute_close_with_retry(asset, value, is_value=True)


    # [ADD TO trade_execution_engine.py]

    # ==================================================================
    # SECTION 3: ATOMIC QUANTITY METHODS (New Refactor)
    # ==================================================================

    def execute_atomic_pair_by_qty(self, leg1_dir: str, leg1_asset: str, leg1_qty: float, 
                                   leg2_dir: str, leg2_asset: str, leg2_qty: float, 
                                   leverage: float) -> Dict[str, Any]:
        """
        Executes a pair trade (Open) by QUANTITY with Transaction Integrity.
        """
        print(f"[EXEC ENGINE] Atomic Open Qty: {leg1_dir} {leg1_asset} {leg1_qty} / {leg2_dir} {leg2_asset} {leg2_qty}")
        
        transaction_state = {
            "leg1": {"executed": False, "order_id": None},
            "leg2": {"executed": False, "order_id": None},
            "rollback": {"attempted": False, "successful": False}
        }

        # --- STEP 1: EXECUTE LEG 1 ---
        leg1_res = self._execute_with_retry(
            asset=leg1_asset, direction=leg1_dir, size=leg1_qty, 
            leverage=leverage, is_value=False
        )
        
        if leg1_res['success']:
            transaction_state['leg1']['executed'] = True
            transaction_state['leg1']['order_id'] = leg1_res['order_id']
            print(f"[EXEC ENGINE] ✅ Leg 1 ({leg1_asset}) Filled")
        else:
            print(f"[EXEC ENGINE] ❌ Leg 1 Failed. Aborting Transaction.")
            return {"success": False, "message": f"Leg 1 Failed: {leg1_res['message']}", "state": transaction_state}

        # --- STEP 2: EXECUTE LEG 2 ---
        leg2_res = self._execute_with_retry(
            asset=leg2_asset, direction=leg2_dir, size=leg2_qty, 
            leverage=leverage, is_value=False
        )
        
        if leg2_res['success']:
            transaction_state['leg2']['executed'] = True
            transaction_state['leg2']['order_id'] = leg2_res['order_id']
            print(f"[EXEC ENGINE] ✅ Atomic Open Qty Complete.")
            return {
                "success": True, 
                "message": f"Atomic Success: {leg1_asset}/{leg2_asset} Opened", 
                "data": transaction_state
            }
        
        # --- STEP 3: ROLLBACK (If Leg 2 Failed) ---
        print(f"[EXEC ENGINE] ⚠️ Leg 2 Failed! Rolling back Leg 1...")
        transaction_state['rollback']['attempted'] = True
        
        rollback_dir = 'short' if leg1_dir.lower() == 'long' else 'long'
        
        rollback_res = self._execute_with_retry(
            asset=leg1_asset, direction=rollback_dir, size=leg1_qty, 
            leverage=leverage, is_value=False
        )
        
        if rollback_res['success']:
            transaction_state['rollback']['successful'] = True
            return {"success": False, "message": "Transaction Failed (Safe Rollback Performed)", "state": transaction_state}
        else:
            return {"success": False, "message": "CRITICAL: Transaction Failed & Rollback Failed", "state": transaction_state}

    def execute_atomic_close_by_qty(self, leg1_asset: str, leg1_qty: float, 
                                    leg2_asset: str, leg2_qty: float) -> Dict[str, Any]:
        """
        Executes a pair CLOSE by QUANTITY with Smart Clamping.
        If requested qty > available qty, it clamps to available (Full Close).
        """
        print(f"[EXEC ENGINE] Atomic Close Qty: {leg1_asset} {leg1_qty} / {leg2_asset} {leg2_qty}")
        
        try:
            positions = self.market_adapter.get_current_positions()
        except Exception as e:
            return {"success": False, "message": f"Failed to fetch positions: {e}"}

        transaction_state = {
            "leg1": {"closed": False},
            "leg2": {"closed": False}
        }

        # --- STEP 1: CLOSE LEG 1 (Smart Qty) ---
        leg1_res = self._execute_smart_qty_close(leg1_asset, leg1_qty, positions.get(leg1_asset))
        
        if leg1_res['success']:
            transaction_state['leg1']['closed'] = True
        else:
            return {"success": False, "message": f"Leg 1 Close Failed: {leg1_res['message']}", "state": transaction_state}

        # --- STEP 2: CLOSE LEG 2 (Smart Qty) ---
        leg2_res = self._execute_smart_qty_close(leg2_asset, leg2_qty, positions.get(leg2_asset))
        
        if leg2_res['success']:
            transaction_state['leg2']['closed'] = True
            print(f"[EXEC ENGINE] ✅ Atomic Close Qty Complete.")
            return {
                "success": True, 
                "message": f"Atomic Close Success: {leg1_asset}/{leg2_asset}", 
                "data": transaction_state
            }
        
        print(f"[EXEC ENGINE] ⚠️ Leg 2 Close Failed! Partial State.")
        return {
            "success": False, 
            "message": "PARTIAL CLOSE ERROR: Leg 1 closed, Leg 2 failed.", 
            "state": transaction_state
        }

    def _execute_smart_qty_close(self, asset, requested_qty, position_data):
        """Helper: Clamps requested quantity to available size if needed."""
        if not position_data:
            print(f"[EXEC ENGINE] Smart Qty: {asset} position not found. Skipping (Success).")
            return {"success": True, "message": "Position already closed"}
        
        available_size = abs(float(position_data.get('size', 0)))
        
        if available_size == 0:
            return {"success": True, "message": "Position size is 0"}

        # Smart Clamp: If we ask for 10 but have 9.9, close 9.9
        if requested_qty > available_size:
            print(f"[EXEC ENGINE] Smart Qty Clamp {asset}: Req {requested_qty} > Avail {available_size}. Closing Full.")
            close_qty = available_size
        else:
            close_qty = requested_qty

        return self._execute_close_with_retry(asset, close_qty, is_value=False)

    def _execute_with_retry(self, asset, direction, size, leverage, is_value=True) -> Dict:
        last_msg = ""
        for attempt in range(1, self.max_retry_attempts + 1):
            if attempt > 1:
                time.sleep(self.retry_backoff_base * (2 ** (attempt - 2)))
                print(f"[EXEC ENGINE] Retry {attempt}/{self.max_retry_attempts} for {asset}...")

            if is_value:
                res = self.market_adapter.execute_market_order_by_value(
                    coin=asset, direction=direction, leverage=leverage, contract_value=size, confirm_order=False
                )
            else:
                res = self.market_adapter.execute_market_order_by_qty(
                    coin=asset, direction=direction, leverage=leverage, quantity=size, confirm_order=False
                )

            if res.get('success'):
                return {"success": True, "order_id": res.get('order_details', {}).get('order_id')}
            last_msg = res.get('message', 'Unknown Error')
        return {"success": False, "message": last_msg}

    def _execute_close_with_retry(self, asset, size, is_value=True) -> Dict:
        last_msg = ""
        for attempt in range(1, self.max_retry_attempts + 1):
            if attempt > 1:
                time.sleep(self.retry_backoff_base * (2 ** (attempt - 2)))
            
            if is_value:
                res = self.market_adapter.execute_close_by_value(
                    coin=asset, target_value=size, confirm_order=False
                )
            else:
                res = self.market_adapter.execute_close_order(
                    coin=asset, close_size=size, confirm_order=False
                )

            if res.get('success'):
                return {"success": True, "order_id": res.get('order_details', {}).get('order_id')}
            last_msg = res.get('message', 'Unknown Error')
        return {"success": False, "message": last_msg}