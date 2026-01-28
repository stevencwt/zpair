import time
from datetime import datetime, timezone

class HitAndRunProfit:
    """
    Hit-and-Run Profit Strategy - Simplified & Production-Ready
    Takes profit at fixed dollar or percentage targets.
    """

    def __init__(self, config=None):
        self.config = config if config else {}

    def analyze_exit_profit(self, position, marketstate, config):
        """
        Analyze if profit targets are met (Modular API Compatible)
        
        Logic:
        1. Check HARD DOLLAR TARGET ($5.33) first
        2. Check Percent Target based on TOTAL INVESTED CAPITAL
        
        Args:
            position: Single position object from modular_decision_engine
            marketstate: Current market state (not used in simplified version)
            config: Trading configuration
            
        Returns:
            Exit signal dict if target met, None otherwise
        """
        
        # 1. Get Strategy Config
        strat_config = config.get('take_profit_strategies', {}).get('hit_and_run_profit', {})
        target_usd = float(strat_config.get('dollar_target', 5.33))
        target_pct = float(strat_config.get('percent_target', 0.5))
        
        # 2. Validate Position
        if not position:
            return None
        
        # 3. Extract PnL directly from position (simplified approach)
        total_pnl = float(position.get('pnl', 0.0))
        
        # 4. Calculate Total Invested Capital from legs
        total_invested_capital = 0.0
        
        if 'legs' in position and isinstance(position['legs'], list):
            for leg in position['legs']:
                total_invested_capital += float(leg.get('value_usd', 0.0))
        else:
            # Fallback: estimate from position sizing config
            size_cfg = config.get('position_sizing', {}).get('sizing_params', {}).get('htf_beta', {})
            base_usd = float(size_cfg.get('base_position_usd', 1000.0))
            total_invested_capital = base_usd * 2  # Pair trading = 2x

        # Safety fallback for denominator
        if total_invested_capital <= 0:
            total_invested_capital = 2000.0

        # 5. Calculate ROI percentage
        current_pct = (total_pnl / total_invested_capital) * 100.0

        # ==========================================
        # PRIORITY CHECK 1: HARD DOLLAR TARGET
        # ==========================================
        if total_pnl >= target_usd:
            return {
                "aggregate_id": position.get("id", position.get("aggregate_id")),
                "exit_type": "take_profit",
                "action": "take_profit",
                "reason": f"Dollar Target Met: ${total_pnl:.2f} >= ${target_usd:.2f}",
                "confidence": 1.0,
                "priority": "high"
            }

        # ==========================================
        # CHECK 2: PERCENTAGE TARGET
        # ==========================================
        if current_pct >= target_pct:
            return {
                "aggregate_id": position.get("id", position.get("aggregate_id")),
                "exit_type": "take_profit",
                "action": "take_profit",
                "reason": f"Percent Target Met: {current_pct:.2f}% >= {target_pct:.2f}% (${total_pnl:.2f})",
                "confidence": 1.0,
                "priority": "high"
            }

        # 6. Debug logging when targets not met
        dist_usd = target_usd - total_pnl
        dist_pct = target_pct - current_pct
        
        print(f"[HIT_RUN_PROFIT] Current PnL: ${total_pnl:.2f} ({current_pct:.2f}%)")
        print(f"[HIT_RUN_PROFIT] Targets: ${target_usd:.2f} ({target_pct:.2f}%)")
        print(f"[HIT_RUN_PROFIT] Distance: ${dist_usd:.2f} ({dist_pct:.2f}%) to target")

        return None


def get_strategy_id():
    """Identifier for the strategy engine"""
    return "hit_and_run_profit"