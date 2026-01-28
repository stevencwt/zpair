#!/usr/bin/env python3
"""
Hyperliquid Market Orders Adapter
Reusable interface for Hyperliquid perpetual trading operations
Extracted from market order scripts for modularity

Requirements:
- hyperliquid-python-sdk
- eth-account
- requests
"""

import json
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    from hyperliquid.info import Info
    from hyperliquid.exchange import Exchange
    from hyperliquid.utils import constants
    from eth_account import Account
    import requests
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install: pip3 install hyperliquid-python-sdk eth-account requests")
    raise ImportError("Missing required dependencies")

class HyperMarketAdapter:
    """
    Hyperliquid Market Adapter for perpetual trading operations
    Handles market order opening, position closing, and order management
    """
    
    def __init__(self, private_key_hex: str, wallet_address: str):
        """Initialize with proper SDK setup for perpetual trading on mainnet"""
        self.private_key_hex = private_key_hex
        self.wallet_address = wallet_address
        
        # Remove 0x prefix for SDK
        self.private_key_clean = self.private_key_hex.replace('0x', '') if self.private_key_hex.startswith('0x') else self.private_key_hex
        
        print(f"Initializing perpetual account for: {self.wallet_address}")
        print("Using MAINNET - real money at risk!")
        
        # Set up mainnet URL
        self.base_url = constants.MAINNET_API_URL
        self.info = None
        self.exchange = None
        self.asset_map = {}
        self._http = requests.Session()
        
        self._initialize_clients()
        
    def _initialize_clients(self):
        """Initialize Info and Exchange clients"""
        try:
            print("Initializing Info client...")
            self.info = Info(self.base_url)
            meta = self.info.meta()
            if meta:
                print(f"Info client connected successfully")
                print(f"   Available perpetual assets: {len(meta.get('universe', []))}")
                for idx, asset in enumerate(meta.get('universe', [])):
                    if 'name' in asset:
                        self.asset_map[asset['name']] = idx
            else:
                print("Info client connected but no meta data received")
        except Exception as e:
            print(f"Error initializing Info client: {e}")
            raise e
        
        try:
            print("Initializing Exchange client...")
            account = Account.from_key(self.private_key_hex)
            self.exchange = Exchange(account, base_url=self.base_url, account_address=self.wallet_address)
            print(f"Exchange client initialized successfully!")
            print(f"   API wallet: {account.address}")
            print(f"   Trading account: {self.wallet_address}")
        except Exception as e:
            print(f"Exchange initialization failed: {e}")
            print("Continuing with Info-only operations")
        
        print(f"HyperMarketAdapter initialized successfully")

    def clean_price_input(self, price_str: str) -> float:
        """Clean and convert price input to float, removing $ and commas"""
        try:
            cleaned = price_str.replace('$', '').replace(',', '').strip()
            return float(cleaned)
        except ValueError:
            raise ValueError(f"Invalid price format: {price_str}")

    def clean_size_input(self, size_str: str) -> float:
        """Clean and convert size input to float, removing commas"""
        try:
            cleaned = size_str.replace(',', '').strip()
            return float(cleaned)
        except ValueError:
            raise ValueError(f"Invalid size format: {size_str}")

    def get_current_prices(self, coins: List[str]) -> Dict[str, float]:
        """Get current prices for specified coins"""
        try:
            all_mids = self.info.all_mids()
            prices = {}
            if isinstance(all_mids, dict):
                for coin in coins:
                    if coin in all_mids:
                        price = all_mids[coin]
                        prices[coin] = float(price) if price else 0
            return prices
        except Exception as e:
            print(f"Error getting current prices: {e}")
            return {}

    def get_available_assets(self) -> List[str]:
        """Get list of available trading assets"""
        try:
            meta = self.info.meta()
            if not meta or not meta.get('universe'):
                return []
            return [asset['name'] for asset in meta['universe'] if 'name' in asset]
        except Exception as e:
            print(f"Error getting available assets: {e}")
            return []

    def get_size_decimals(self, coin: str) -> int:
        """Get size decimals for a specific coin"""
        try:
            meta = self.info.meta()
            if meta and 'universe' in meta:
                for asset in meta['universe']:
                    if asset.get('name') == coin:
                        return asset.get('szDecimals', 5)
            return 5
        except Exception as e:
            print(f"Error getting size decimals: {e}")
            return 5

    def validate_and_round_size(self, coin: str, size: float) -> float:
        """Validate and properly round size according to exchange requirements"""
        try:
            sz_decimals = self.get_size_decimals(coin)
            
            # Round to the specified decimals
            rounded_size = round(size, sz_decimals)
            
            # Check minimum size requirement
            min_size = 10 ** (-sz_decimals + 1)
            if rounded_size < min_size:
                raise ValueError(f"Size {rounded_size} is below minimum {min_size} for {coin}")
            
            # Additional validation: ensure the rounded size makes sense
            if abs(rounded_size - size) / size > 0.01:  # More than 1% difference
                print(f"Warning: Significant rounding difference for {coin}")
                print(f"   Original: {size}")
                print(f"   Rounded: {rounded_size}")
                print(f"   Difference: {abs(rounded_size - size) / size * 100:.2f}%")
            
            return rounded_size
            
        except Exception as e:
            raise ValueError(f"Error validating size for {coin}: {e}")

    def execute_market_order_by_value(self, coin: str, direction: str, leverage: float, contract_value: float, confirm_order: bool = True):
        """Execute a market order by contract value"""
        print(f"\n=== EXECUTE MARKET ORDER BY VALUE ===")
        print(f"Asset: {coin}, Direction: {direction.upper()}, Value: ${contract_value}, Leverage: {leverage}x")
        
        if not self.exchange:
            result = {"status": "error", "message": "Exchange client not available"}
            print(f"ERROR: {result['message']}")
            return result
        
        try:
            if direction.lower() not in ['long', 'short']:
                result = {"status": "error", "message": "Invalid direction"}
                print(f"ERROR: {result['message']}")
                return result
            is_buy = direction.lower() == 'long'
            if leverage <= 0 or leverage > 50:
                result = {"status": "error", "message": "Leverage must be between 0 and 50"}
                print(f"ERROR: {result['message']}")
                return result
            if contract_value <= 0:
                result = {"status": "error", "message": "Contract value must be positive"}
                print(f"ERROR: {result['message']}")
                return result
            
            meta = self.info.meta()
            if not meta or not meta.get('universe'):
                result = {"status": "error", "message": "Unable to retrieve available assets"}
                print(f"ERROR: {result['message']}")
                return result
            available_coins = [asset['name'] for asset in meta['universe'] if 'name' in asset]
            if coin.upper() not in available_coins:
                result = {"status": "error", "message": f"Invalid coin: {coin}"}
                print(f"ERROR: {result['message']}")
                return result
            coin = coin.upper()
            
            current_price = self.get_current_prices([coin]).get(coin, 0)
            if current_price <= 0:
                result = {"status": "error", "message": f"Unable to retrieve current price for {coin}"}
                print(f"ERROR: {result['message']}")
                return result
            
            size = contract_value / current_price
            margin = contract_value / leverage
            
            print(f"Current price: ${current_price:.4f}")
            print(f"Calculated size: {size:.6f}")
            print(f"Required margin: ${margin:.2f}")
            
            try:
                size_rounded = self.validate_and_round_size(coin, size)
                print(f"Rounded size: {size_rounded}")
            except ValueError as e:
                result = {"status": "error", "message": str(e)}
                print(f"ERROR: {result['message']}")
                return result
            
            order_info = {
                "coin": coin,
                "direction": direction.upper(),
                "leverage": leverage,
                "contract_value": contract_value,
                "margin": margin,
                "size": size_rounded,
                "current_price": current_price
            }
            
            print(f"Placing {direction.upper()} market order for {coin}:")
            print(f"   Leverage: {leverage}x")
            print(f"   Contract Value: ${contract_value:,.2f}")
            print(f"   Size: {size_rounded} contracts @ ${current_price:,.4f}")
            print(f"   Margin Used: ${margin:,.2f}")
            
            if confirm_order:
                confirm = input("\nConfirm order placement? (y/N): ").lower()
                if confirm != 'y':
                    result = {"status": "cancelled", "message": "Order cancelled by user"}
                    print(f"ORDER CANCELLED: {result['message']}")
                    return result
            
            print("Setting leverage...")
            leverage_result = self.exchange.update_leverage(leverage=int(leverage), name=coin)
            if leverage_result.get('status') != 'ok':
                result = {"status": "error", "message": "Failed to set leverage"}
                print(f"ERROR: {result['message']}")
                return result
            print(f"SUCCESS: Leverage set to {leverage}x for {coin}")
            
            print("Placing market order...")
            order_result = self.exchange.market_open(coin, is_buy, size_rounded)
            success = order_result.get('status') == 'ok'
            
            order_status = "unknown"
            order_details = {}
            if success:
                print("SUCCESS: Market order placed!")
                statuses = order_result.get('response', {}).get('data', {}).get('statuses', [])
                for status in statuses:
                    if 'filled' in status:
                        filled = status['filled']
                        order_status = "filled"
                        order_details = {
                            "order_id": filled.get('oid'),
                            "filled_size": filled.get('totalSz'),
                            "avg_price": filled.get('avgPx')
                        }
                        print(f"ORDER FILLED: Size {order_details['filled_size']}, Avg Price ${float(order_details['avg_price']):.4f}")
                    elif 'resting' in status:
                        order_status = "resting"
                        order_details = {"order_id": status['resting'].get('oid')}
                        print(f"ORDER RESTING: ID {order_details['order_id']}")
                    elif 'error' in status:
                        order_status = "error"
                        order_details = {"error": status['error']}
                        print(f"ORDER ERROR: {order_details['error']}")
            else:
                order_status = "failed"
                error_info = order_result.get('response', {}).get('error', 'Unknown error')
                order_details = {"error": error_info}
                print(f"ORDER FAILED: {error_info}")
            
            result = {
                "status": order_status,
                "order_info": order_info,
                "order_details": order_details,
                "success": success
            }
            
            print(f"FINAL RESULT: {order_status.upper()} - Success: {success}")
            return result
            
        except Exception as e:
            print(f"EXCEPTION in execute_market_order_by_value: {e}")
            return {"status": "error", "message": str(e)}

    def execute_market_order_by_qty(self, coin: str, direction: str, leverage: float, quantity: float, confirm_order: bool = True):
        """Execute a market order by quantity"""
        print(f"\n=== EXECUTE MARKET ORDER BY QUANTITY ===")
        print(f"Asset: {coin}, Direction: {direction.upper()}, Quantity: {quantity}, Leverage: {leverage}x")
        
        if not self.exchange:
            result = {"status": "error", "message": "Exchange client not available"}
            print(f"ERROR: {result['message']}")
            return result
        
        try:
            if direction.lower() not in ['long', 'short']:
                result = {"status": "error", "message": "Invalid direction"}
                print(f"ERROR: {result['message']}")
                return result
            is_buy = direction.lower() == 'long'
            if leverage <= 0 or leverage > 50:
                result = {"status": "error", "message": "Leverage must be between 0 and 50"}
                print(f"ERROR: {result['message']}")
                return result
            if quantity <= 0:
                result = {"status": "error", "message": "Quantity must be positive"}
                print(f"ERROR: {result['message']}")
                return result
            
            meta = self.info.meta()
            if not meta or not meta.get('universe'):
                result = {"status": "error", "message": "Unable to retrieve available assets"}
                print(f"ERROR: {result['message']}")
                return result
            available_coins = [asset['name'] for asset in meta['universe'] if 'name' in asset]
            if coin.upper() not in available_coins:
                result = {"status": "error", "message": f"Invalid coin: {coin}"}
                print(f"ERROR: {result['message']}")
                return result
            coin = coin.upper()
            
            current_price = self.get_current_prices([coin]).get(coin, 0)
            if current_price <= 0:
                result = {"status": "error", "message": f"Unable to retrieve current price for {coin}"}
                print(f"ERROR: {result['message']}")
                return result
            
            print(f"Current price: ${current_price:.4f}")
            
            try:
                size_rounded = self.validate_and_round_size(coin, quantity)
                print(f"Validated size: {size_rounded}")
            except ValueError as e:
                result = {"status": "error", "message": str(e)}
                print(f"ERROR: {result['message']}")
                return result
            
            contract_value = size_rounded * current_price
            margin = contract_value / leverage
            
            print(f"Contract value: ${contract_value:.2f}")
            print(f"Required margin: ${margin:.2f}")
            
            order_info = {
                "coin": coin,
                "direction": direction.upper(),
                "leverage": leverage,
                "contract_value": contract_value,
                "margin": margin,
                "size": size_rounded,
                "current_price": current_price
            }
            
            print(f"Placing {direction.upper()} market order for {coin}:")
            print(f"   Leverage: {leverage}x")
            print(f"   Quantity: {size_rounded} contracts")
            print(f"   Contract Value: ${contract_value:,.2f} @ ${current_price:,.4f}")
            print(f"   Margin Used: ${margin:,.2f}")
            
            if confirm_order:
                confirm = input("\nConfirm order placement? (y/N): ").lower()
                if confirm != 'y':
                    result = {"status": "cancelled", "message": "Order cancelled by user"}
                    print(f"ORDER CANCELLED: {result['message']}")
                    return result
            
            print("Setting leverage...")
            leverage_result = self.exchange.update_leverage(leverage=int(leverage), name=coin)
            if leverage_result.get('status') != 'ok':
                result = {"status": "error", "message": "Failed to set leverage"}
                print(f"ERROR: {result['message']}")
                return result
            print(f"SUCCESS: Leverage set to {leverage}x for {coin}")
            
            print("Placing market order...")
            order_result = self.exchange.market_open(coin, is_buy, size_rounded)
            success = order_result.get('status') == 'ok'
            
            order_status = "unknown"
            order_details = {}
            if success:
                print("SUCCESS: Market order placed!")
                statuses = order_result.get('response', {}).get('data', {}).get('statuses', [])
                for status in statuses:
                    if 'filled' in status:
                        filled = status['filled']
                        order_status = "filled"
                        order_details = {
                            "order_id": filled.get('oid'),
                            "filled_size": filled.get('totalSz'),
                            "avg_price": filled.get('avgPx')
                        }
                        print(f"ORDER FILLED: Size {order_details['filled_size']}, Avg Price ${float(order_details['avg_price']):.4f}")
                    elif 'resting' in status:
                        order_status = "resting"
                        order_details = {"order_id": status['resting'].get('oid')}
                        print(f"ORDER RESTING: ID {order_details['order_id']}")
                    elif 'error' in status:
                        order_status = "error"
                        order_details = {"error": status['error']}
                        print(f"ORDER ERROR: {order_details['error']}")
            else:
                order_status = "failed"
                error_info = order_result.get('response', {}).get('error', 'Unknown error')
                order_details = {"error": error_info}
                print(f"ORDER FAILED: {error_info}")
            
            result = {
                "status": order_status,
                "order_info": order_info,
                "order_details": order_details,
                "success": success
            }
            
            print(f"FINAL RESULT: {order_status.upper()} - Success: {success}")
            return result
            
        except Exception as e:
            print(f"EXCEPTION in execute_market_order_by_qty: {e}")
            return {"status": "error", "message": str(e)}

    def get_current_positions(self) -> Dict[str, Any]:
        """Get current open positions"""
        try:
            user_state = self.info.user_state(self.wallet_address)
            positions = {}
            
            if user_state and 'assetPositions' in user_state:
                for pos in user_state['assetPositions']:
                    if 'position' in pos and 'coin' in pos['position']:
                        coin = pos['position']['coin']
                        position_data = pos['position']
                        
                        # Only include positions with non-zero size
                        size = float(position_data.get('szi', 0))
                        if size != 0:
                            positions[coin] = {
                                'size': size,
                                'entry_px': float(position_data.get('entryPx', 0)),
                                'pnl': float(position_data.get('unrealizedPnl', 0)),
                                'leverage': float(position_data.get('leverage', {}).get('value', 0)),
                                'margin_used': float(position_data.get('marginUsed', 0)),
                                'side': 'LONG' if size > 0 else 'SHORT'
                            }
            
            return positions
        except Exception as e:
            print(f"Error getting current positions: {e}")
            return {}

    def validate_close_by_smart_value(self, coin: str, target_value: float):
        """Validate smart close by contract value without executing"""
        positions = self.get_current_positions()
        if coin not in positions:
            return {"valid": False, "error": f"No open position found for {coin}"}
        
        position = positions[coin]
        current_size = abs(position['size'])
        current_price = self.get_current_prices([coin]).get(coin, 0)
        
        if current_price <= 0:
            return {"valid": False, "error": f"Unable to retrieve current price for {coin}"}
        
        max_contract_value = current_size * current_price
        if target_value > max_contract_value:
            return {"valid": False, "error": f"Contract value {target_value} exceeds maximum of ${max_contract_value:.2f}"}
        
        # Check minimum contract value requirement (Hyperliquid requires $10 minimum)
        if target_value < 10.0:
            return {"valid": False, "error": f"Contract value ${target_value:.2f} is below minimum $10.00 required by exchange"}
        
        # Calculate equivalent quantity
        equivalent_qty = target_value / current_price
        sz_decimals = self.get_size_decimals(coin)
        
        # Smart close logic
        percentage_of_position = equivalent_qty / current_size
        
        if equivalent_qty >= current_size or percentage_of_position >= 0.95:
            # Close all if equivalent qty equals or exceeds available qty, or if 95%+ of position
            close_size = current_size
            actual_contract_value = close_size * current_price
        else:
            # Close exact contract value amount
            close_size = round(equivalent_qty, sz_decimals)
            actual_contract_value = close_size * current_price
            
            # Validate minimum size
            min_size = 10 ** (-sz_decimals + 1)
            if close_size < min_size:
                return {"valid": False, "error": f"Calculated size {close_size} is below minimum {min_size}"}
            
            # Re-check minimum contract value after rounding
            if actual_contract_value < 10.0:
                return {"valid": False, "error": f"Actual contract value ${actual_contract_value:.2f} after rounding is below minimum $10.00"}
        
        return {"valid": True, "close_size": close_size, "actual_contract_value": actual_contract_value}

    def validate_close_by_qty(self, coin: str, quantity: float):
        """Validate close by quantity without executing"""
        positions = self.get_current_positions()
        if coin not in positions:
            return {"valid": False, "error": f"No open position found for {coin}"}
        
        position = positions[coin]
        current_size = abs(position['size'])
        current_price = self.get_current_prices([coin]).get(coin, 0)
        
        if current_price <= 0:
            return {"valid": False, "error": f"Unable to retrieve current price for {coin}"}
        
        if quantity <= 0:
            return {"valid": False, "error": "Close size must be positive"}
        
        if quantity > current_size:
            return {"valid": False, "error": f"Close size {quantity} exceeds position size {current_size}"}
        
        # Validate size decimals and minimum
        sz_decimals = self.get_size_decimals(coin)
        close_size_rounded = round(quantity, sz_decimals)
        min_size = 10 ** (-sz_decimals + 1)
        if close_size_rounded < min_size:
            return {"valid": False, "error": f"Rounded size {close_size_rounded} is below minimum {min_size}"}
        
        # Check minimum contract value requirement (Hyperliquid requires $10 minimum)
        contract_value = close_size_rounded * current_price
        if contract_value < 10.0:
            return {"valid": False, "error": f"Contract value ${contract_value:.2f} is below minimum $10.00 required by exchange"}
        
        return {"valid": True, "close_size": close_size_rounded, "contract_value": contract_value}

    def execute_close_order(self, coin: str, close_size: float, confirm_order: bool = True):
        """Execute a market order to close position"""
        print(f"\n=== EXECUTE CLOSE ORDER ===")
        print(f"Asset: {coin}, Close Size: {close_size}")
        
        if not self.exchange:
            result = {"status": "error", "message": "Exchange client not available"}
            print(f"ERROR: {result['message']}")
            return result
        
        try:
            # Get current positions
            positions = self.get_current_positions()
            if coin not in positions:
                result = {"status": "error", "message": f"No open position found for {coin}"}
                print(f"ERROR: {result['message']}")
                return result
            
            position = positions[coin]
            current_size = abs(position['size'])
            position_side = position['side']
            
            print(f"Current position: {position_side} {current_size}")
            print(f"Position entry price: ${position['entry_px']:.4f}")
            print(f"Current PnL: ${position['pnl']:.2f}")
            
            if close_size <= 0:
                result = {"status": "error", "message": "Close size must be positive"}
                print(f"ERROR: {result['message']}")
                return result
            
            if close_size > current_size:
                result = {"status": "error", "message": f"Close size {close_size} exceeds position size {current_size}"}
                print(f"ERROR: {result['message']}")
                return result
            
            # To close a position, we need to place an order in the opposite direction
            # Long position: place sell order (is_buy = False)
            # Short position: place buy order (is_buy = True)
            is_buy = position_side == 'SHORT'
            close_direction = 'BUY' if is_buy else 'SELL'
            
            # Get current price for display
            current_price = self.get_current_prices([coin]).get(coin, 0)
            if current_price <= 0:
                result = {"status": "error", "message": f"Unable to retrieve current price for {coin}"}
                print(f"ERROR: {result['message']}")
                return result
            
            print(f"Current market price: ${current_price:.4f}")
            
            # Get size decimals for proper rounding
            sz_decimals = self.get_size_decimals(coin)
            close_size_rounded = round(close_size, sz_decimals)
            
            print(f"Rounded close size: {close_size_rounded}")
            
            order_info = {
                "coin": coin,
                "position_side": position_side,
                "close_direction": close_direction,
                "close_size": close_size_rounded,
                "current_size": current_size,
                "remaining_size": current_size - close_size_rounded,
                "current_price": current_price,
                "entry_price": position['entry_px'],
                "current_pnl": position['pnl'],
                "contract_value": close_size_rounded * current_price
            }
            
            print(f"Closing {position_side} position for {coin}:")
            print(f"   Current Position Size: {current_size}")
            print(f"   Close Size: {close_size_rounded}")
            print(f"   Remaining Size: {current_size - close_size_rounded}")
            print(f"   Close Direction: {close_direction}")
            print(f"   Current Price: ${current_price:,.4f}")
            print(f"   Entry Price: ${position['entry_px']:,.4f}")
            print(f"   Contract Value: ${close_size_rounded * current_price:,.2f}")
            print(f"   Current PnL: ${position['pnl']:,.2f}")
            
            if confirm_order:
                confirm = input("\nConfirm position close? (y/N): ").lower()
                if confirm != 'y':
                    result = {"status": "cancelled", "message": "Order cancelled by user"}
                    print(f"ORDER CANCELLED: {result['message']}")
                    return result
            
            print("Executing close order...")
            # Execute the closing order using market_open with opposite direction (same as original script)
            order_result = self.exchange.market_open(coin, is_buy, close_size_rounded)
            success = order_result.get('status') == 'ok'
            
            order_status = "unknown"
            order_details = {}
            if success:
                print("SUCCESS: Market close order placed!")
                statuses = order_result.get('response', {}).get('data', {}).get('statuses', [])
                for status in statuses:
                    if 'filled' in status:
                        filled = status['filled']
                        order_status = "filled"
                        order_details = {
                            "order_id": filled.get('oid'),
                            "filled_size": filled.get('totalSz'),
                            "avg_price": filled.get('avgPx')
                        }
                        print(f"POSITION CLOSED: Size {order_details['filled_size']}, Avg Price ${float(order_details['avg_price']):.4f}")
                    elif 'resting' in status:
                        order_status = "resting"
                        order_details = {"order_id": status['resting'].get('oid')}
                        print(f"CLOSE ORDER RESTING: ID {order_details['order_id']}")
                    elif 'error' in status:
                        order_status = "error"
                        order_details = {"error": status['error']}
                        print(f"CLOSE ORDER ERROR: {order_details['error']}")
            else:
                order_status = "failed"
                error_info = order_result.get('response', {}).get('error', 'Unknown error')
                order_details = {"error": error_info}
                print(f"CLOSE ORDER FAILED: {error_info}")
            
            result = {
                "status": order_status,
                "order_info": order_info,
                "order_details": order_details,
                "success": success
            }
            
            print(f"CLOSE RESULT: {order_status.upper()} - Success: {success}")
            return result
            
        except Exception as e:
            print(f"EXCEPTION in execute_close_order: {e}")
            return {"status": "error", "message": str(e)}

    def execute_close_by_value(self, coin: str, target_value: float, confirm_order: bool = True):
        """Execute close by contract value"""
        print(f"\n=== EXECUTE CLOSE BY VALUE ===")
    def execute_close_by_value(self, coin: str, target_value: float, confirm_order: bool = True):
        """Execute close by contract value"""
        print(f"\n=== EXECUTE CLOSE BY VALUE ===")
        print(f"Asset: {coin}, Target Value: ${target_value:.2f}")
        
        positions = self.get_current_positions()
        if coin not in positions:
            result = {"status": "error", "message": f"No open position found for {coin}"}
            print(f"ERROR: {result['message']}")
            return result
        
        position = positions[coin]
        current_size = abs(position['size'])
        current_price = self.get_current_prices([coin]).get(coin, 0)
        
        if current_price <= 0:
            result = {"status": "error", "message": f"Unable to retrieve current price for {coin}"}
            print(f"ERROR: {result['message']}")
            return result
        
        max_contract_value = current_size * current_price
        print(f"Current position size: {current_size}")
        print(f"Current price: ${current_price:.4f}")
        print(f"Maximum contract value: ${max_contract_value:.2f}")
        
        if target_value > max_contract_value:
            result = {"status": "error", "message": f"Contract value {target_value} exceeds maximum of ${max_contract_value:.2f}"}
            print(f"ERROR: {result['message']}")
            return result
        
        # Calculate equivalent quantity
        equivalent_qty = target_value / current_price
        sz_decimals = self.get_size_decimals(coin)
        close_size = round(equivalent_qty, sz_decimals)
        
        print(f"Calculated close size: {close_size}")
        
        # Validate minimum size
        min_size = 10 ** (-sz_decimals + 1)
        if close_size < min_size:
            result = {"status": "error", "message": f"Calculated size {close_size} is below minimum {min_size}"}
            print(f"ERROR: {result['message']}")
            return result
        
        # Check minimum contract value requirement
        actual_contract_value = close_size * current_price
        if actual_contract_value < 10.0:
            result = {"status": "error", "message": f"Actual contract value ${actual_contract_value:.2f} is below minimum $10.00"}
            print(f"ERROR: {result['message']}")
            return result
        
        print(f"Actual contract value to close: ${actual_contract_value:.2f}")
        
        return self.execute_close_order(coin, close_size, confirm_order)

    def execute_close_by_smart_value(self, coin: str, target_value: float, confirm_order: bool = True):
        """Execute smart close by contract value with 95% threshold logic"""
        print(f"\n=== EXECUTE SMART CLOSE BY VALUE ===")
        print(f"Asset: {coin}, Target Value: ${target_value:.2f}")
        
        positions = self.get_current_positions()
        if coin not in positions:
            result = {"status": "error", "message": f"No open position found for {coin}"}
            print(f"ERROR: {result['message']}")
            return result
        
        position = positions[coin]
        current_size = abs(position['size'])
        current_price = self.get_current_prices([coin]).get(coin, 0)
        
        if current_price <= 0:
            result = {"status": "error", "message": f"Unable to retrieve current price for {coin}"}
            print(f"ERROR: {result['message']}")
            return result
        
        max_contract_value = current_size * current_price
        print(f"Current position size: {current_size}")
        print(f"Current price: ${current_price:.4f}")
        print(f"Maximum contract value: ${max_contract_value:.2f}")
        
        if target_value > max_contract_value:
            result = {"status": "error", "message": f"Contract value {target_value} exceeds maximum of ${max_contract_value:.2f}"}
            print(f"ERROR: {result['message']}")
            return result
        
        # Calculate equivalent quantity
        equivalent_qty = target_value / current_price
        sz_decimals = self.get_size_decimals(coin)
        
        # Smart close logic
        percentage_of_position = equivalent_qty / current_size
        
        print(f"Smart close analysis for {coin}:")
        print(f"   Target contract value: ${target_value:.2f}")
        print(f"   Equivalent quantity: {equivalent_qty:.6f}")
        print(f"   Percentage of position: {percentage_of_position * 100:.2f}%")
        
        if equivalent_qty >= current_size or percentage_of_position >= 0.95:
            # Close all if equivalent qty equals or exceeds available qty, or if 95%+ of position
            close_size = current_size
            close_type = "CLOSE ALL"
            
            if equivalent_qty >= current_size:
                reason = "equivalent quantity equals or exceeds available quantity"
            else:
                reason = "equivalent quantity is 95% or more of available quantity"
            
            print(f"   Decision: {close_type} ({reason})")
        else:
            # Close exact contract value amount
            close_size = round(equivalent_qty, sz_decimals)
            
            # Validate minimum size
            min_size = 10 ** (-sz_decimals + 1)
            if close_size < min_size:
                result = {"status": "error", "message": f"Calculated size {close_size} is below minimum {min_size}"}
                print(f"ERROR: {result['message']}")
                return result
            
            close_type = "EXACT CONTRACT VALUE"
            print(f"   Decision: {close_type} (less than 95% of position)")
        
        print(f"   Close quantity: {close_size}")
        
        return self.execute_close_order(coin, close_size, confirm_order)

    def print_order_result(self, result: Dict[str, Any], prefix: str = "Order"):
        """Print formatted order result"""
        print(f"\n=== {prefix.upper()} EXECUTION SUMMARY ===")
        print(f"Status: {result.get('status', 'unknown').upper()}")
        if result.get('success') and result.get('status') not in ['error', 'failed']:
            print("RESULT: Order executed successfully!")
            order_details = result.get('order_details', {})
            if 'filled_size' in order_details:
                print(f"Filled Size: {order_details['filled_size']}")
            if 'avg_price' in order_details:
                print(f"Average Price: ${float(order_details['avg_price']):.4f}")
        else:
            error_msg = result.get('message', result.get('order_details', {}).get('error', 'Unknown error'))
            print(f"RESULT: Order failed - {error_msg}")
        print("=" * 50)

    def print_close_result(self, result: Dict[str, Any], prefix: str = "Close"):
        """Print formatted close result"""
        print(f"\n=== {prefix.upper()} SUMMARY ===")
        print(f"Status: {result.get('status', 'unknown').upper()}")
        if result.get('success') and result.get('status') not in ['error', 'failed']:
            print("RESULT: Position closed successfully!")
            order_details = result.get('order_details', {})
            if 'filled_size' in order_details:
                print(f"Filled Size: {order_details['filled_size']}")
            if 'avg_price' in order_details:
                print(f"Average Price: ${float(order_details['avg_price']):.4f}")
        else:
            error_msg = result.get('message', result.get('order_details', {}).get('error', 'Unknown error'))
            print(f"RESULT: Close failed - {error_msg}")
        print("=" * 50)

    def show_positions(self):
        """Display current positions"""
        print("\n" + "="*80)
        print("CURRENT POSITIONS")
        print("="*80)
        
        positions = self.get_current_positions()
        if not positions:
            print("No open positions found.")
            print("="*80)
            return
        
        # Get current prices for all positions
        coins = list(positions.keys())
        current_prices = self.get_current_prices(coins)
        
        print(f"{'Coin':<8} {'Side':<6} {'Size':<12} {'Entry':<10} {'Current':<10} {'Contract Value':<15} {'PnL':<12} {'Leverage':<8}")
        print("-" * 80)
        
        total_pnl = 0
        for coin, pos in positions.items():
            current_price = current_prices.get(coin, 0)
            contract_value = abs(pos['size']) * current_price
            total_pnl += pos['pnl']
            print(f"{coin:<8} {pos['side']:<6} {abs(pos['size']):<12.4f} ${pos['entry_px']:<9.4f} ${current_price:<9.4f} ${contract_value:<14.2f} ${pos['pnl']:<11.2f} {pos['leverage']:<7.1f}x")
        
        print("-" * 80)
        print(f"TOTAL POSITIONS: {len(positions)} | TOTAL PnL: ${total_pnl:.2f}")
        print("="*80)