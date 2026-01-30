#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modular Decision Engine - Strategy Coordination and Selection
ENHANCED VERSION: Context-Specific Trading Signal Analysis Implementation + Dual Scaling Mode Support
UNIFIED VERSION: Always uses aggregate_id for position references
UPDATED VERSION: Take-Profit Strategy Restructuring with Modular Imports
REFACTORED VERSION: Removed hardcoded HTFBiasLTFTimingWithScaling, added modular extreme_scaling_strategy

NEW FEATURES:
- 3-mode context detection (entry, exit_cutloss, exit_profit)
- Context-specific analysis fields with detailed reasoning
- Pipe-delimited analysis showing thresholds, progress, and conditions
- Backward compatibility with existing reasoning field
- Modular take-profit strategies with separate module imports
- Fully modular scaling strategies (simple and extreme)
- ENHANCED: Dual scaling mode contextual messaging support

ARCHITECTURE CHANGES:
- Removed hardcoded HTFBiasLTFTimingWithScaling class
- Added modular ExtremeScalingStrategy from separate module
- Maintained SimpleScalingStrategy for basic scaling
- All scaling strategies now follow modular pattern
- ENHANCED: Added dual scaling mode message generation
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

# Import entry and cut-loss strategies from main trading_strategies module
from trading_strategies import (
    HTFBiasLTFTimingEntry,
    MultiCriteriaStopLoss,
    PositionScalingStrategy
)

# UPDATED: Import modular take-profit strategies from separate modules
from hit_and_run_profit import HitAndRunProfit
from let_profit_run_profit import LetProfitRun
from trailing_stop_profit import TrailingStopProfit

# UPDATED: Import modular cut-loss scaling strategies from separate modules
from simple_scaling_strategy import SimpleScalingStrategy
from extreme_scaling_strategy import ExtremeScalingStrategy

# PHASE 3B: Import single-asset and dual-hybrid strategies
try:
    from hurst_rsi_entry_strategy import HurstRSIEntry
    from fixed_target_exit_strategy import FixedTargetExit
    from stop_loss_scaling_exit_strategy import StopLossScalingExit
    SINGLE_ASSET_STRATEGIES_AVAILABLE = True
except ImportError as e:
    print(f"[MODULAR] Single asset strategies not available: {e}")
    SINGLE_ASSET_STRATEGIES_AVAILABLE = False

try:
    from dual_hybrid_entry_strategy import DualHybridEntry
    from alternating_scaling_exit_strategy import AlternatingScalingExit
    DUAL_HYBRID_STRATEGIES_AVAILABLE = True
except ImportError as e:
    print(f"[MODULAR] Dual hybrid strategies not available: {e}")
    DUAL_HYBRID_STRATEGIES_AVAILABLE = False



# Import aggregator for unified system
from position_aggregator import PositionAggregator


class ModularDecisionEngine:
    """
    Coordinates modular strategies with unified position management.
    Always uses aggregate_id for position references.
    ENHANCED: Context-specific trading signal analysis with detailed reasoning.
    UPDATED: Modular take-profit strategy support with separate module imports.
    REFACTORED: Fully modular scaling strategy architecture.
    ENHANCED: Dual scaling mode contextual messaging support.
    """
    
    def __init__(self, config: Dict[str, Any], trading_mode: Optional[Dict[str, Any]] = None):
        self.config = config
        self.last_decision_time = 0.0
        self.decision_cooldown = 3.0
        
        # PHASE 3: Store trading mode for mode-specific strategy selection
        self.trading_mode = trading_mode or {"name": "pairs_trading"}
        self.mode_name = self.trading_mode.get("name", "pairs_trading")
        
        # PHASE 3: Mode-specific strategy registries
        self._initialize_strategy_registries()
        
        #These are now defined inside _initialize_strate, hence these lines below are removed:
        # Entry strategy registry (unchanged)
        #self.ENTRY_STRATEGIES = {
        #    "htf_bias_ltf_timing": HTFBiasLTFTimingEntry,
        #}
        #
        # UPDATED: Take-profit strategy registry with modular imports
        #self.TAKE_PROFIT_STRATEGIES = {
        #    "hit_and_run_profit": HitAndRunProfit,
        #    "let_profit_run_profit": LetProfitRun,
        #    "trailing_stop_profit": TrailingStopProfit,
        #}
        #
        # UPDATED: Cut-loss strategy registry - fully modular
        #self.CUT_LOSS_STRATEGIES = {
        #    "multi_criteria_stop": MultiCriteriaStopLoss,
        #    "simple_scaling_strategy": SimpleScalingStrategy,
        #    "extreme_scaling_strategy": ExtremeScalingStrategy,
        #}
        
        # Load active strategies
        self.active_entry = self._load_strategy("entry", self.ENTRY_STRATEGIES)
        self.active_profit = self._load_strategy("take_profit", self.TAKE_PROFIT_STRATEGIES)
        self.active_loss = self._load_strategy("cut_loss", self.CUT_LOSS_STRATEGIES)
        
        # Check active_loss for scaling instead of active_entry
        self.is_scaling_strategy = isinstance(self.active_loss, PositionScalingStrategy)
        
        # Always initialize position aggregator for unified system
        self.position_aggregator = PositionAggregator()
        
        print(f"[MODULAR] Unified modular decision engine initialized with context-specific analysis and fully modular strategies")
        print(f"[MODULAR] Trading mode: {self.mode_name}")
        print(f"[MODULAR] Active entry strategy: {self._get_active_strategy_name('entry')}")
        print(f"[MODULAR] Active profit strategy: {self._get_active_strategy_name('take_profit')}")
        print(f"[MODULAR] Active loss strategy: {self._get_active_strategy_name('cut_loss')}")
        print(f"[MODULAR] Strategy mode: {'Scaling' if self.is_scaling_strategy else 'Single-position'}")
        print(f"[MODULAR] Signal format: aggregate_id (unified) with context-specific analysis")
        print(f"[MODULAR] Architecture: Fully modular (all strategies in separate modules)")
        print(f"[MODULAR] Enhanced: Dual scaling mode contextual messaging support")
    
    def _load_strategy(self, category: str, registry: Dict[str, Any]):
        """Load strategy based on configuration with enhanced error handling"""
        strategy_name = self.config.get("active_strategies", {}).get(category)
        strategy_class = registry.get(strategy_name)
        
        if strategy_class:
            print(f"[MODULAR] Loading {category} strategy: {strategy_name}")
            try:
                return strategy_class(self.config)
            except Exception as e:
                print(f"[MODULAR ERROR] Failed to initialize {category} strategy '{strategy_name}': {e}")
                return None
        else:
            if registry:
                default_name = list(registry.keys())[0]
                default_class = list(registry.values())[0]
                print(f"[MODULAR] Strategy '{strategy_name}' not found, using default: {default_name}")
                try:
                    return default_class(self.config)
                except Exception as e:
                    print(f"[MODULAR ERROR] Failed to initialize default {category} strategy '{default_name}': {e}")
                    return None
            else:
                print(f"[MODULAR ERROR] No {category} strategies available")
                return None

    def _initialize_strategy_registries(self):
        """
        Initialize strategy registries based on trading mode.
        PHASE 3: Mode-aware strategy selection
        """
        if self.mode_name == "pairs_trading":
            # Existing pairs trading strategies
            self.ENTRY_STRATEGIES = {
                "htf_bias_ltf_timing": HTFBiasLTFTimingEntry,
            }
            
            self.TAKE_PROFIT_STRATEGIES = {
                "hit_and_run_profit": HitAndRunProfit,
                "let_profit_run_profit": LetProfitRun,
                "trailing_stop_profit": TrailingStopProfit,
            }
            
            self.CUT_LOSS_STRATEGIES = {
                "multi_criteria_stop": MultiCriteriaStopLoss,
                "simple_scaling_strategy": SimpleScalingStrategy,
                "extreme_scaling_strategy": ExtremeScalingStrategy,
            }
            
        elif self.mode_name == "single_asset":
            # PHASE 3B: Single asset strategies
            if SINGLE_ASSET_STRATEGIES_AVAILABLE:
                self.ENTRY_STRATEGIES = {
                    "hurst_rsi_entry": HurstRSIEntry,
                }
                self.TAKE_PROFIT_STRATEGIES = {
                    "fixed_target_exit": FixedTargetExit,
                }
                self.CUT_LOSS_STRATEGIES = {
                    "stop_loss_scaling_exit": StopLossScalingExit,
                }
                print(f"[MODULAR] Single asset strategies loaded successfully")
            else:
                print(f"[MODULAR] Single asset strategies not available - fallback to pairs")
                self.ENTRY_STRATEGIES = {"htf_bias_ltf_timing": HTFBiasLTFTimingEntry}
                self.TAKE_PROFIT_STRATEGIES = {"hit_and_run_profit": HitAndRunProfit}
                self.CUT_LOSS_STRATEGIES = {"simple_scaling_strategy": SimpleScalingStrategy}
            
        elif self.mode_name == "dual_hybrid":
            # PHASE 3B: Dual hybrid strategies (placeholder)
            if DUAL_HYBRID_STRATEGIES_AVAILABLE:
                self.ENTRY_STRATEGIES = {
                    "dual_hybrid_entry": DualHybridEntry,
                }
                self.TAKE_PROFIT_STRATEGIES = {
                    "let_profit_run_profit": LetProfitRun,  # Reuse pairs strategy
                }
                self.CUT_LOSS_STRATEGIES = {
                    "alternating_scaling_exit": AlternatingScalingExit,
                }
                print(f"[MODULAR] Dual hybrid strategies loaded (PLACEHOLDER)")
            else:
                print(f"[MODULAR] Dual hybrid strategies not available - fallback to pairs")
                self.ENTRY_STRATEGIES = {"htf_bias_ltf_timing": HTFBiasLTFTimingEntry}
                self.TAKE_PROFIT_STRATEGIES = {"let_profit_run_profit": LetProfitRun}
                self.CUT_LOSS_STRATEGIES = {"simple_scaling_strategy": SimpleScalingStrategy}
    
    def _get_active_strategy_name(self, category: str) -> str:
        """Get the name of the currently active strategy for a category"""
        configured_name = self.config.get("active_strategies", {}).get(category)
        
        if category == "entry":
            registry = self.ENTRY_STRATEGIES
        elif category == "take_profit":
            registry = self.TAKE_PROFIT_STRATEGIES
        else:
            registry = self.CUT_LOSS_STRATEGIES
        
        if configured_name in registry:
            return configured_name
        elif registry:
            return list(registry.keys())[0]
        else:
            return "none"
    
    def detect_trading_context(self, marketstate: Dict[str, Any]) -> str:
        """
        Determine the current trading context (ENTRY, EXIT_CUTLOSS, EXIT_PROFIT).
        ROBUST FIX: Correctly reads P&L from Unified Portfolio to route to Profit mode.
        """
        # 1. Get Portfolio Data
        portfolio = marketstate.get('portfolio', {})
        positions = portfolio.get('positions', [])
        
        # 2. If no positions -> ENTRY mode
        if not positions:
            # print("[CONTEXT] No positions detected - context: ENTRY")
            return 'entry'

        # 3. Calculate Total P&L
        # We sum up P&L from all positions to be safe
        total_pnl = sum(float(p.get('pnl', 0.0)) for p in positions)
        
        # 4. Determine Context based on P&L
        if total_pnl > 0:
            # print(f"[CONTEXT] Position in PROFIT (${total_pnl:.2f}) - context: EXIT_PROFIT")
            return 'exit_profit'
        else:
            # print(f"[CONTEXT] Position in LOSS/BE (${total_pnl:.2f}) - context: EXIT_CUTLOSS")
            return 'exit_cutloss'

    def generate_entry_analysis(self, marketstate: Dict[str, Any]) -> str:
        """
        Enhanced entry analysis with direction control information and HTF-LTF conflict detection.
        NEW: Now includes volatility sync health check for comprehensive regime assessment.
        """
        try:
            # Get z-scores and correlation data
            regression = marketstate.get("regression", {})
            regression_htf = marketstate.get("regression_htf", {})
            
            ltf_zscore = regression.get("zscore", 0.0)
            htf_zscore = regression_htf.get("zscore", 0.0)
            correlation = regression.get("correlation", 0.0)
            
            # Get entry strategy configuration
            entry_config = self.config.get("entry_strategies", {})
            active_entry_name = self._get_active_strategy_name("entry")
            strategy_config = entry_config.get(active_entry_name, {})
            
            analysis_parts = []
            
            # HTF Bias analysis
            htf_config = strategy_config.get("htf_bias_config", {})
            strong_positive = htf_config.get("strong_positive_threshold", 1.0)
            strong_negative = htf_config.get("strong_negative_threshold", -1.0)
            neutral_band = htf_config.get("neutral_band", [-0.5, 0.5])
            
            if htf_zscore >= strong_positive:
                htf_status = f"HTF STRONG POSITIVE: {htf_zscore:.2f} >= {strong_positive}"
            elif htf_zscore <= strong_negative:
                htf_status = f"HTF STRONG NEGATIVE: {htf_zscore:.2f} <= {strong_negative}"
            elif neutral_band[0] <= htf_zscore <= neutral_band[1]:
                htf_status = f"HTF NEUTRAL: {htf_zscore:.2f} in [{neutral_band[0]}, {neutral_band[1]}]"
            else:
                htf_status = f"HTF MODERATE: {htf_zscore:.2f}"
            
            analysis_parts.append(htf_status)
            
            # LTF Timing analysis
            ltf_config = strategy_config.get("ltf_timing_config", {})
            entry_thresholds = ltf_config.get("entry_thresholds", {})
            long_threshold = entry_thresholds.get("long", -1.5)
            short_threshold = entry_thresholds.get("short", 1.5)
            
            if ltf_zscore <= long_threshold:
                ltf_status = f"LTF LONG signal: {ltf_zscore:.2f} <= {long_threshold}"
            elif ltf_zscore >= short_threshold:
                ltf_status = f"LTF SHORT signal: {ltf_zscore:.2f} >= {short_threshold}"
            else:
                ltf_status = f"LTF no signal: {ltf_zscore:.2f} (need <= {long_threshold} or >= {short_threshold})"
            
            analysis_parts.append(ltf_status)
            
            # Direction control analysis
            direction_control = strategy_config.get("direction_control", {})
            allowed_directions = direction_control.get("allowed_directions", "both")
            
            if allowed_directions == "long_only":
                analysis_parts.append("Direction: LONG ONLY")
            elif allowed_directions == "short_only":
                analysis_parts.append("Direction: SHORT ONLY")
            else:
                analysis_parts.append("Direction: BOTH allowed")
            
            # Correlation health check using actual strategy thresholds
            regime_requirements = strategy_config.get("market_regime_requirements", {})
            min_correlation = regime_requirements.get("min_correlation", 0.15)
            
            if abs(correlation) < min_correlation:
                analysis_parts.append(f"⚠️ LOW correlation: {abs(correlation):.3f} < {min_correlation}")
            else:
                analysis_parts.append(f"Correlation OK: {abs(correlation):.3f} >= {min_correlation}")
            
            # NEW: Volatility sync health check
            regime_assessment = marketstate.get("regime_assessment", {})
            volatility_sync = regime_assessment.get("volatility_sync", {}).get("ratio", 0.0)
            min_volatility_sync = regime_requirements.get("min_volatility_sync", 0.4)
            
            if volatility_sync < min_volatility_sync:
                analysis_parts.append(f"❌ VOLATILITY SYNC FAIL: {volatility_sync:.3f} < {min_volatility_sync} (1m TF)")
            else:
                analysis_parts.append(f"Volatility Sync OK: {volatility_sync:.3f} >= {min_volatility_sync}")
            
            return " | ".join(analysis_parts)
            
        except Exception as e:
            print(f"[ANALYSIS ERROR] Failed to generate entry analysis: {e}")
            return f"Entry analysis failed: {str(e)}"

    def _detect_scaling_mode(self, strategy_result: Optional[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """Extract scaling mode and metadata from strategy response"""
        if not strategy_result:
            return "legacy", {}
        
        metadata = strategy_result.get("add_position_metadata", {})
        scaling_mode = metadata.get("scaling_mode", "legacy")
        return scaling_mode, metadata

    def _generate_fixed_increment_message(self, current_zscore: float, metadata: Dict[str, Any], threshold_met: bool, direction: str) -> str:
        """Generate fixed increment mode threshold message"""
        calculated_threshold = metadata.get("calculated_threshold")
        entry_zscore = metadata.get("entry_zscore")
        scaling_info = metadata.get("scaling_info", "")
        threshold_description = metadata.get("threshold_description", "")
        
        if calculated_threshold is not None:
            if threshold_met:
                if entry_zscore is not None:
                    return f"Fixed increment threshold MET: {current_zscore:.2f} {'<=' if direction == 'LONG' else '>='} {calculated_threshold:.2f} ({scaling_info})"
                else:
                    return f"Fixed increment threshold MET: {current_zscore:.2f} {'<=' if direction == 'LONG' else '>='} {calculated_threshold:.2f} ({threshold_description})"
            else:
                if entry_zscore is not None:
                    return f"Fixed increment threshold NOT MET: {current_zscore:.2f} {'<' if direction == 'LONG' else '>'} {calculated_threshold:.2f} ({scaling_info})"
                else:
                    return f"Fixed increment threshold NOT MET: {current_zscore:.2f} vs {calculated_threshold:.2f} ({threshold_description})"
        else:
            return f"Fixed increment threshold NOT MET: {current_zscore:.2f} (calculation unavailable)"

    def _generate_percentage_mode_message(self, current_zscore: float, metadata: Dict[str, Any], threshold_met: bool, direction: str) -> str:
        """Generate percentage mode threshold message"""
        calculated_threshold = metadata.get("calculated_threshold")
        percentage_threshold = metadata.get("percentage_threshold", 20.0)
        previous_entry_zscore = metadata.get("previous_entry_zscore")
        
        if calculated_threshold is not None and previous_entry_zscore is not None:
            if threshold_met:
                return f"Percentage threshold MET: {current_zscore:.2f} is {percentage_threshold}% worse than {previous_entry_zscore:.2f}"
            else:
                return f"Percentage threshold NOT MET: {current_zscore:.2f} need {percentage_threshold}% worse than {previous_entry_zscore:.2f}"
        else:
            return f"Percentage threshold NOT MET: {current_zscore:.2f} (calculation unavailable)"

    def _generate_fallback_message(self, current_zscore: float, metadata: Dict[str, Any], threshold_met: bool, direction: str) -> str:
        """Generate fallback mode threshold message"""
        calculated_threshold = metadata.get("calculated_threshold")
        threshold_description = metadata.get("threshold_description", "progressive from base")
        
        if calculated_threshold is not None:
            if threshold_met:
                return f"Fallback threshold MET: {current_zscore:.2f} {'<=' if direction == 'LONG' else '>='} {calculated_threshold:.2f} ({threshold_description})"
            else:
                return f"Fallback threshold NOT MET: {current_zscore:.2f} {'<' if direction == 'LONG' else '>'} {calculated_threshold:.2f} ({threshold_description})"
        else:
            return f"Fallback threshold NOT MET: {current_zscore:.2f} (legacy mode)"

    def _build_enhanced_cutloss_analysis_message(self, position: Dict[str, Any], current_pnl: float, 
                                               strategy_result: Optional[Dict[str, Any]], current_zscore: float) -> str:
        """Enhanced message builder that routes to appropriate scaling mode handler"""
        try:
            aggregated_metrics = position.get('aggregated_metrics', {})
            direction = aggregated_metrics.get('direction', 'UNKNOWN')
            position_count = aggregated_metrics.get('actual_component_count', 1)
            
            # Get strategy configuration
            loss_config = self.config.get("cut_loss_strategies", {})
            active_loss_name = self._get_active_strategy_name("cut_loss")
            strategy_config = loss_config.get(active_loss_name, {})
            scaling_config = strategy_config.get("position_scaling_config", {})
            add_conditions = scaling_config.get("add_conditions", {})
            loss_threshold = add_conditions.get("floating_loss_threshold_usd", 5.0)
            max_add_positions = scaling_config.get("max_add_positions", 4)
            max_total = 1 + max_add_positions
            
            analysis_parts = []
            
            # Loss threshold check (unchanged)
            if current_pnl <= -loss_threshold:
                analysis_parts.append(f"Loss threshold MET: ${current_pnl:.2f} <= -${loss_threshold}")
            else:
                analysis_parts.append(f"Loss threshold NOT MET: ${current_pnl:.2f} > -${loss_threshold}")
            
            # Enhanced threshold analysis based on strategy result
            scaling_mode, metadata = self._detect_scaling_mode(strategy_result)
            threshold_met = strategy_result and strategy_result.get('action') == 'ADD_POSITION'
            
            if scaling_mode == "fixed_increment":
                threshold_message = self._generate_fixed_increment_message(current_zscore, metadata, threshold_met, direction)
                analysis_parts.append(threshold_message)
            elif scaling_mode == "percentage_based":
                threshold_message = self._generate_percentage_mode_message(current_zscore, metadata, threshold_met, direction)
                analysis_parts.append(threshold_message)
            elif scaling_mode == "legacy" and metadata.get("threshold_description") == "fallback_progressive":
                threshold_message = self._generate_fallback_message(current_zscore, metadata, threshold_met, direction)
                analysis_parts.append(threshold_message)
            else:
                # Scaling mode not detected or legacy mode - provide generic message
                if threshold_met:
                    analysis_parts.append(f"Threshold MET: {current_zscore:.2f} (calculated threshold passed)")
                else:
                    analysis_parts.append(f"Threshold NOT MET: {current_zscore:.2f} (awaiting extreme z-score)")
            
            # Enhanced position scaling availability with mode indication
            if position_count < max_total:
                mode_indicator = ""
                if scaling_mode == "fixed_increment":
                    mode_indicator = " - Fixed Increment Mode"
                elif scaling_mode == "percentage_based":
                    percentage = metadata.get("percentage_threshold", 20.0)
                    mode_indicator = f" - Percentage Mode ({percentage}%)"
                elif scaling_mode == "legacy" and metadata.get("threshold_description") == "fallback_progressive":
                    mode_indicator = " - Fallback Mode"
                
                analysis_parts.append(f"Position scaling: AVAILABLE ({position_count}/{max_total}){mode_indicator}")
            else:
                analysis_parts.append(f"Position scaling: BLOCKED ({position_count}/{max_total})")
            
            return " | ".join(analysis_parts)
            
        except Exception as e:
            print(f"[ANALYSIS ERROR] Enhanced cutloss analysis failed: {e}")
            return f"Enhanced cutloss analysis failed: {str(e)}"

    def generate_exit_cutloss_analysis(self, position: Dict[str, Any], marketstate: Dict[str, Any]) -> str:
        """
        NEW: Generate detailed cut-loss exit analysis with current values vs thresholds.
        """
        try:
            #legacy code pnl_tracking = position.get('pnl_tracking', {})
            #legacy code aggregated_metrics = position.get('aggregated_metrics', {})
            pnl_usd = position.get('pnl', 0.0)
            aggregated_metrics = position.get('aggregated_metrics', {})

            position_count = aggregated_metrics.get('actual_component_count', 1)
            direction = aggregated_metrics.get('direction', 'UNKNOWN')
            
            # Get current z-score for scaling decisions
            regression = marketstate.get("regression", {})
            current_zscore = regression.get("zscore", 0.0)
            
            # Get cut-loss thresholds from config
            loss_config = self.config.get("cut_loss_strategies", {})
            active_loss_name = self._get_active_strategy_name("cut_loss")
            strategy_config = loss_config.get(active_loss_name, {})
            
            analysis_parts = []
            
            # Strategy-specific analysis
            if active_loss_name == "extreme_scaling_strategy":
                # Extreme scaling analysis
                scaling_config = strategy_config.get("position_scaling_config", {})
                add_conditions = scaling_config.get("add_conditions", {})
                loss_threshold = add_conditions.get("floating_loss_threshold_usd", 5.0)
                max_add_positions = scaling_config.get("max_add_positions", 3)
                max_total = 1 + max_add_positions
                
                # Loss threshold check
                if pnl_usd <= -loss_threshold:
                    excess_loss = abs(pnl_usd) - loss_threshold
                    analysis_parts.append(f"Loss threshold MET: ${pnl_usd:.2f} <= -${loss_threshold} (${excess_loss:.2f} excess)")
                else:
                    remaining_loss = loss_threshold - abs(pnl_usd)
                    analysis_parts.append(f"Loss threshold NOT MET: ${pnl_usd:.2f} > -${loss_threshold} (${remaining_loss:.2f} to go)")
                
                # Position scaling check
                if position_count < max_total:
                    analysis_parts.append(f"Position scaling: AVAILABLE ({position_count}/{max_total})")
                    
                    # Extreme threshold analysis (simplified)
                    extreme_config = scaling_config.get("extreme_tracking", {})
                    min_long_threshold = extreme_config.get("min_long_threshold", -1.0)
                    min_short_threshold = extreme_config.get("min_short_threshold", 1.0)
                    
                    if direction == "LONG" and current_zscore <= min_long_threshold:
                        analysis_parts.append(f"Extreme LONG threshold MET: {current_zscore:.2f} <= {min_long_threshold}")
                    elif direction == "SHORT" and current_zscore >= min_short_threshold:
                        analysis_parts.append(f"Extreme SHORT threshold MET: {current_zscore:.2f} >= {min_short_threshold}")
                    else:
                        analysis_parts.append(f"Extreme threshold NOT MET: {current_zscore:.2f} vs {direction} requirements")
                else:
                    analysis_parts.append(f"Position scaling: BLOCKED ({position_count}/{max_total} max reached)")
                    
            elif active_loss_name == "simple_scaling_strategy":
                scaling_config = strategy_config.get("position_scaling_config", {})
                add_conditions = scaling_config.get("add_conditions", {})
                loss_threshold = add_conditions.get("floating_loss_threshold_usd", 5.0)
                max_add_positions = scaling_config.get("max_add_positions", 4)
                max_total = 1 + max_add_positions
                
                # Loss threshold check
                if pnl_usd <= -loss_threshold:
                    analysis_parts.append(f"Loss threshold MET: ${pnl_usd:.2f} <= -${loss_threshold}")
                else:
                    analysis_parts.append(f"Loss threshold NOT MET: ${pnl_usd:.2f} > -${loss_threshold}")
                
                # Let strategy handle all threshold logic - no preliminary checks
                simple_thresholds = scaling_config.get("simple_zscore_thresholds", {})
                scaling_mode = simple_thresholds.get("scaling_mode", "fixed_increment")
                
                if position_count < max_total:
                    analysis_parts.append(f"Position scaling: AVAILABLE ({position_count}/{max_total}) - {scaling_mode} mode")
                    analysis_parts.append(f"Z-score threshold: Strategy will evaluate entry-based thresholds")
                else:
                    analysis_parts.append(f"Position scaling: BLOCKED ({position_count}/{max_total})")
                    
            else:
                # Multi-criteria stop or other non-scaling strategy
                stop_config = strategy_config.get("stop_loss_exit", {})
                max_loss_dollar = stop_config.get("max_loss_dollar", 30.0)
                
                if pnl_usd <= -max_loss_dollar:
                    analysis_parts.append(f"Stop loss TRIGGERED: ${pnl_usd:.2f} <= -${max_loss_dollar}")
                else:
                    remaining_buffer = max_loss_dollar - abs(pnl_usd)
                    analysis_parts.append(f"Stop loss NOT triggered: ${pnl_usd:.2f} (${remaining_buffer:.2f} buffer)")
                
                analysis_parts.append("Position scaling: NOT AVAILABLE (non-scaling strategy)")
            
            return " | ".join(analysis_parts)
            
        except Exception as e:
            print(f"[ANALYSIS ERROR] Failed to generate cut-loss analysis: {e}")
            return f"Cut-loss analysis failed: {str(e)}"

    def generate_exit_profit_analysis(self, position: Dict[str, Any], marketstate: Dict[str, Any]) -> str:
        """
        NEW: Generate detailed profit-taking analysis with current values vs thresholds.
        """
        try:
            #pnl_tracking = position.get('pnl_tracking', {})
            #$aggregated_metrics = position.get('aggregated_metrics', {})
            #timestamps = position.get('timestamps', {})
            #pnl_usd = pnl_tracking.get('floating_pnl_usd', 0.0)
            #pnl_percent = pnl_tracking.get('floating_pnl_percent', 0.0)
            # [FIX] UnifiedTradeEngine stores PnL directly in position['pnl']

            pnl_usd = position.get('pnl', 0.0)
            aggregated_metrics = position.get('aggregated_metrics', {})
            timestamps = position.get('timestamps', {})

            # Calculate pnl_percent from position value
            entry_price = position.get('entry_price', 0.0)
            size = position.get('size', 1.0)
            position_value = entry_price * size
            pnl_percent = (pnl_usd / position_value * 100) if position_value > 0 else 0.0


            direction = aggregated_metrics.get('direction', 'UNKNOWN')
            
            # Get current z-score for mean reversion analysis
            regression = marketstate.get("regression", {})
            current_zscore = regression.get("zscore", 0.0)
            
            # Calculate hold duration
            initial_entry = timestamps.get('initial_entry')
            hold_hours = 0.0
            if initial_entry:
                try:
                    entry_time = datetime.fromisoformat(initial_entry.replace('Z', '+00:00'))
                    current_time = datetime.now(timezone.utc)
                    hold_hours = (current_time - entry_time).total_seconds() / 3600
                except:
                    pass
            
            # Get profit-taking thresholds from config
            profit_config = self.config.get("take_profit_strategies", {})
            active_profit_name = self._get_active_strategy_name("take_profit")
            strategy_config = profit_config.get(active_profit_name, {})
            
            analysis_parts = []
            
            # Strategy-specific analysis
            if active_profit_name == "hit_and_run_profit":
                dollar_target = strategy_config.get("dollar_target", 5.0)
                percent_target = strategy_config.get("percent_target", 0.6)
                max_wait_hours = strategy_config.get("max_wait_hours", 5.0)
                
                # Profit targets
                if pnl_usd >= dollar_target:
                    analysis_parts.append(f"Dollar target MET: ${pnl_usd:.2f} >= ${dollar_target}")
                else:
                    remaining_dollar = dollar_target - pnl_usd
                    analysis_parts.append(f"Dollar target NOT MET: ${pnl_usd:.2f} (${remaining_dollar:.2f} to go)")
                
                if pnl_percent >= percent_target:
                    analysis_parts.append(f"Percent target MET: {pnl_percent:.2f}% >= {percent_target}%")
                else:
                    remaining_percent = percent_target - pnl_percent
                    analysis_parts.append(f"Percent target NOT MET: {pnl_percent:.2f}% ({remaining_percent:.2f}% to go)")
                
                # Time constraint
                if hold_hours >= max_wait_hours:
                    analysis_parts.append(f"Time limit REACHED: {hold_hours:.1f}h >= {max_wait_hours}h")
                else:
                    remaining_time = max_wait_hours - hold_hours
                    analysis_parts.append(f"Time remaining: {remaining_time:.1f}h of {max_wait_hours}h")
                
            elif active_profit_name == "trailing_stop_profit":
                activation_dollar = strategy_config.get("activation_dollar_target", 3.0)
                trailing_distance = strategy_config.get("trailing_distance_usd", 1.0)
                
                if pnl_usd >= activation_dollar:
                    analysis_parts.append(f"Trailing ACTIVE: ${pnl_usd:.2f} >= ${activation_dollar} (trail: ${trailing_distance})")
                else:
                    remaining_activation = activation_dollar - pnl_usd
                    analysis_parts.append(f"Trailing NOT active: ${pnl_usd:.2f} (${remaining_activation:.2f} to activation)")
                
                # Mean reversion check
                reversion_threshold = 0.3
                if abs(current_zscore) <= reversion_threshold:
                    analysis_parts.append(f"Mean reversion signal: |{current_zscore:.2f}| <= {reversion_threshold}")
                else:
                    analysis_parts.append(f"No reversion: |{current_zscore:.2f}| > {reversion_threshold}")
                    
            else:
                # Let profit run or other strategy
                min_dollar_target = strategy_config.get("min_dollar_target", 5.0)
                min_percent_target = strategy_config.get("min_percent_target", 1.0)
                
                if pnl_usd >= min_dollar_target:
                    analysis_parts.append(f"Min profit MET: ${pnl_usd:.2f} >= ${min_dollar_target}")
                else:
                    analysis_parts.append(f"Min profit NOT MET: ${pnl_usd:.2f} < ${min_dollar_target}")
                
                # Mean reversion factor
                if direction == "LONG" and current_zscore >= 0:
                    analysis_parts.append(f"Mean reversion: LONG position with positive zscore {current_zscore:.2f}")
                elif direction == "SHORT" and current_zscore <= 0:
                    analysis_parts.append(f"Mean reversion: SHORT position with negative zscore {current_zscore:.2f}")
                else:
                    analysis_parts.append(f"No reversion signal: {direction} with zscore {current_zscore:.2f}")
            
            return " | ".join(analysis_parts)
            
        except Exception as e:
            print(f"[ANALYSIS ERROR] Failed to generate profit analysis: {e}")
            return f"Profit analysis failed: {str(e)}"

    def make_trading_decision(self, marketstate: Dict[str, Any]) -> Dict[str, Any]:
        """
        ENHANCED: Decision making with context-specific analysis generation and modular take-profit strategies.
        ENHANCED: Dual scaling mode contextual messaging support.
        """
        current_time = time.time()
        if current_time - self.last_decision_time < self.decision_cooldown:
            return self._create_wait_decision("Decision cooldown active")
        
        try:
            if not self._validate_marketstate(marketstate):
                return self._create_wait_decision("Invalid market state structure")
            
            # NEW: Detect trading context
            context = self.detect_trading_context(marketstate)
            print(f"[MODULAR] Detected trading context: {context.upper()}")
            
            # Always load aggregated position (unified approach)
            position = self._get_aggregated_position(marketstate)
            
            # Initialize analysis fields
            entry_analysis = None
            exit_cutloss_analysis = None
            exit_profit_analysis = None
            
            # Generate context-specific analysis and process signals
            if context == 'entry':
                # Generate entry analysis
                entry_analysis = self.generate_entry_analysis(marketstate)
                print(f"[MODULAR] Entry analysis: {entry_analysis}")
                
                # Process entry logic
                if self.active_entry:
                    entry_signal = self.active_entry.analyze_entry(marketstate, self.config)
                    if entry_signal:
                        # Generate aggregate_id for new entry
                        entry_signal['aggregate_id'] = self._generate_aggregate_id()
                        self.last_decision_time = current_time
                        print(f"[MODULAR] Entry strategy generated ENTRY signal")
                        return self._create_entry_decision(entry_signal, marketstate, context, 
                                                         entry_analysis, exit_cutloss_analysis, exit_profit_analysis)
                
                return self._create_wait_decision("No entry conditions met", context, 
                                                entry_analysis, exit_cutloss_analysis, exit_profit_analysis)
            
            elif context == 'exit_cutloss':
                # ENHANCED: Process cut-loss logic first to get strategy result
                loss_exit = None
                if self.active_loss:
                    loss_exit = self.active_loss.analyze_exit_loss(position, marketstate, self.config)
                
                # ENHANCED: Generate enhanced cut-loss analysis with strategy result
                if self._get_active_strategy_name("cut_loss") == "simple_scaling_strategy":
                    # Use enhanced analysis for simple scaling strategy
                    #pnl_tracking = position.get('pnl_tracking', {}) if position else {}
                    #current_pnl = pnl_tracking.get('floating_pnl_usd', 0.0)
                    current_pnl = position.get('pnl', 0.0) if position else 0.0
                    regression = marketstate.get("regression", {})
                    current_zscore = regression.get("zscore", 0.0)
                    exit_cutloss_analysis = self._build_enhanced_cutloss_analysis_message(position, current_pnl, loss_exit, current_zscore)
                else:
                    # Use original analysis for other strategies
                    exit_cutloss_analysis = self.generate_exit_cutloss_analysis(position, marketstate)
                
                print(f"[MODULAR] Cut-loss analysis: {exit_cutloss_analysis}")
                
                # Handle strategy result if available
                if loss_exit:
                    if 'aggregate_id' not in loss_exit:
                        loss_exit['aggregate_id'] = position['aggregate_id']
                    
                    self.last_decision_time = current_time
                    
                    # Handle different types of exit signals
                    if loss_exit.get('action') == 'ADD_POSITION':
                        print(f"[MODULAR] Loss strategy generated ADD_POSITION signal")
                        return self._create_add_position_decision(loss_exit, marketstate, context,
                                                                entry_analysis, exit_cutloss_analysis, exit_profit_analysis)
                    else:
                        print(f"[MODULAR] Loss strategy generated EXIT signal")
                        return self._create_exit_decision([loss_exit], context,
                                                        entry_analysis, exit_cutloss_analysis, exit_profit_analysis)
                
                return self._create_wait_decision("Cut-loss conditions not met", context,
                                                entry_analysis, exit_cutloss_analysis, exit_profit_analysis)
            
            elif context == 'exit_profit':
                # Generate profit analysis
                exit_profit_analysis = self.generate_exit_profit_analysis(position, marketstate)
                print(f"[MODULAR] Profit analysis: {exit_profit_analysis}")
                
                # Process profit-taking logic using modular strategies
                if self.active_profit:
                    profit_exit = self.active_profit.analyze_exit_profit(position, marketstate, self.config)
                    if profit_exit:
                        if 'aggregate_id' not in profit_exit:
                            #profit_exit['aggregate_id'] = position['aggregate_id']
                            # [RECTIFIED] Robust ID Access (Supports both Legacy 'id' and Unified 'aggregate_id')
                            profit_exit['aggregate_id'] = position.get('aggregate_id', position.get('id', 'unknown_id'))
                        self.last_decision_time = current_time
                        print(f"[MODULAR] Profit strategy '{self._get_active_strategy_name('take_profit')}' generated EXIT signal")
                        return self._create_exit_decision([profit_exit], context,
                                                        entry_analysis, exit_cutloss_analysis, exit_profit_analysis)
                
                return self._create_wait_decision("Profit-taking conditions not met", context,
                                                entry_analysis, exit_cutloss_analysis, exit_profit_analysis)
            
            else:
                return self._create_wait_decision(f"Unknown context: {context}")
                
        except Exception as e:
            print(f"[MODULAR ERROR] Decision making failed: {e}")
            import traceback
            traceback.print_exc()
            return self._create_wait_decision(f"Decision error: {str(e)}")
    
    def _validate_marketstate(self, marketstate: Dict[str, Any]) -> bool:
        """Basic validation of marketstate structure"""
        required_sections = ["regression", "regression_htf", "asof"]
        
        for section in required_sections:
            if section not in marketstate:
                print(f"[MODULAR ERROR] Missing required marketstate section: {section}")
                return False
        
        return True
    
    def _get_aggregated_position(self, marketstate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract aggregated position from Unified Market State."""
        try:
            # 1. Look for the new Unified Key first
            if 'portfolio' in marketstate:
                positions = marketstate['portfolio'].get('positions', [])
                if positions:
                    return positions[0]
            
            # 2. Fallback to Legacy (if 'unified_trade_engine' added the compatibility layer)
            # This ensures we don't break if you switch back to old engine temporarily
            trade_simulation = marketstate.get("trade_simulation", {})
            if trade_simulation:
                 # ... existing legacy logic ...
                 pass

            return None
            
        except Exception as e:
            print(f"[MODULAR ERROR] Failed to extract position: {e}")
            return None
    
    def _validate_add_signal(self,
        signal: Dict[str, Any],
        position: Optional[Dict[str, Any]]
    ) -> Tuple[bool, List[str]]:
        """Validate ADD_POSITION signal consistency using aggregate_id."""
        errors = []
        
        if not position:
            errors.append("ADD_POSITION signal but no existing position")
            return False, errors
        
        # Check aggregate_id matches
        signal_agg_id = signal.get('aggregate_id', '')
        position_agg_id = position.get('aggregate_id', '')
        
        if signal_agg_id != position_agg_id:
            errors.append(f"Aggregate ID mismatch: {signal_agg_id} vs {position_agg_id}")
        
        # Check direction matches
        signal_direction = signal.get('direction', '')
        position_direction = position.get('aggregated_metrics', {}).get('direction', '')
        
        if signal_direction != position_direction:
            errors.append(f"Direction mismatch: {signal_direction} vs {position_direction}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def _generate_aggregate_id(self) -> str:
        """Generate aggregate_id for new positions"""
        timestamp = datetime.now(timezone.utc)
        return f"AGG-NEW-{timestamp.strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
    
    def _create_entry_decision(self,
        entry_signal: Dict[str, Any],
        marketstate: Dict[str, Any],
        analysis_context: str,
        entry_analysis: Optional[str],
        exit_cutloss_analysis: Optional[str],
        exit_profit_analysis: Optional[str]
    ) -> Dict[str, Any]:
        """Create entry trading decision with context-specific analysis"""
        regression = marketstate.get("regression", {})
        prices = marketstate.get("prices", {})
        indicators = marketstate.get("indicators", {})
        
        # Add execution context for trade executor
        execution_context = {
            "price_a": prices.get("primary", 100.0),
            "price_b": prices.get("reference", 1.0),
            "hedge_ratio": 0.5
        }
        
        # Select active analysis for backward compatibility
        active_analysis = entry_analysis or entry_signal.get("reasoning", "Entry conditions met")
        
        return {
            "current_recommendation": {
                "recommendation": "ENTER",
                "direction": entry_signal["direction"],
                "confidence": entry_signal["confidence"],
                "risk_level": "MEDIUM",
                
                # NEW: Context-specific analysis fields
                "analysis_context": analysis_context,
                "entry_analysis": entry_analysis,
                "exit_cutloss_analysis": exit_cutloss_analysis,
                "exit_profit_analysis": exit_profit_analysis,
                
                # Backward compatibility
                "reasoning": active_analysis,
                
                "strategy_used": entry_signal["strategy"],
                "position_size_modifier": 1.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "execution_context": execution_context,
                
                # Entry signals use aggregate_id for unified position management
                "aggregate_id": entry_signal["aggregate_id"],
                "signal_strength": abs(entry_signal.get("ltf_signal", 0.0)),
                "key_factors": [
                    f"Strategy: {entry_signal['strategy']}",
                    f"HTF bias: {entry_signal.get('htf_bias', 0.0):.3f}",
                    f"LTF signal: {entry_signal.get('ltf_signal', 0.0):.3f}",
                    active_analysis
                ],
                "execution_thresholds": {
                    "entry_signal": "ltf_zscore",
                    "entry_value": entry_signal.get("ltf_signal", 0.0),
                    "htf_bias": entry_signal.get("htf_bias", 0.0)
                },
                "price_ref": prices,
                "zscore": regression.get("zscore", 0.0),
                "rsi": indicators.get("rsi", 50.0)
            }
        }
    
    def _create_add_position_decision(self,
        add_signal: Dict[str, Any],
        marketstate: Dict[str, Any],
        analysis_context: str,
        entry_analysis: Optional[str],
        exit_cutloss_analysis: Optional[str],
        exit_profit_analysis: Optional[str]
    ) -> Dict[str, Any]:
        """Create ADD_POSITION trading decision with context-specific analysis"""
        regression = marketstate.get("regression", {})
        prices = marketstate.get("prices", {})
        indicators = marketstate.get("indicators", {})
        
        # Add execution context for trade executor
        execution_context = {
            "price_a": prices.get("primary", 100.0),
            "price_b": prices.get("reference", 1.0),
            "hedge_ratio": 0.5
        }
        
        # Select active analysis for backward compatibility
        active_analysis = exit_cutloss_analysis or add_signal.get("reasoning", "Add position signal")
        
        return {
            "current_recommendation": {
                "recommendation": "ADD_POSITION",
                "direction": add_signal["direction"],
                "confidence": add_signal["confidence"],
                "risk_level": "MEDIUM",
                
                # NEW: Context-specific analysis fields
                "analysis_context": analysis_context,
                "entry_analysis": entry_analysis,
                "exit_cutloss_analysis": exit_cutloss_analysis,
                "exit_profit_analysis": exit_profit_analysis,
                
                # Backward compatibility
                "reasoning": active_analysis,
                
                "strategy_used": add_signal["strategy"],
                "position_size_modifier": 1.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "execution_context": execution_context,  
                
                # ADD_POSITION specific fields - all use aggregate_id
                "aggregate_id": add_signal["aggregate_id"],
                "add_position_metadata": add_signal.get("add_position_metadata", {}),
                "threshold_met": add_signal.get("threshold_met", 0.0),
                
                # Standard fields
                "signal_strength": abs(add_signal.get("ltf_signal", 0.0)),
                "key_factors": [
                    f"Strategy: {add_signal['strategy']}",
                    f"Threshold crossed: {add_signal.get('threshold_met', 0.0):.3f}",
                    f"Add-on signal from exit strategy",
                    active_analysis
                ],
                "execution_thresholds": {
                    "threshold_crossed": add_signal.get('threshold_met', 0.0),
                    "htf_bias": add_signal.get('htf_bias', 0.0),
                    "ltf_signal": add_signal.get('ltf_signal', 0.0)
                },
                "price_ref": prices,
                "zscore": regression.get("zscore", 0.0),
                "rsi": indicators.get("rsi", 50.0)
            }
        }
    
    def _create_exit_decision(self, 
        exit_signals: List[Dict[str, Any]], 
        analysis_context: str,
        entry_analysis: Optional[str],
        exit_cutloss_analysis: Optional[str],
        exit_profit_analysis: Optional[str]
    ) -> Dict[str, Any]:
        """Create exit trading decision with context-specific analysis"""
        primary_exit = exit_signals[0] if exit_signals else {}
        
        # Ensure all exit signals have aggregate_id
        processed_exit_signals = []
        for signal in exit_signals:
            if 'aggregate_id' not in signal:
                print(f"[MODULAR WARNING] Exit signal missing aggregate_id: {signal}")
            processed_exit_signals.append(signal)
        
        # Select active analysis for backward compatibility
        if analysis_context == 'exit_cutloss':
            active_analysis = exit_cutloss_analysis or primary_exit.get("reason", "Cut-loss exit conditions met")
        elif analysis_context == 'exit_profit':
            active_analysis = exit_profit_analysis or primary_exit.get("reason", "Profit-taking conditions met")
        else:
            active_analysis = primary_exit.get("reason", "Exit conditions met")
        
        return {
            "current_recommendation": {
                "recommendation": "EXIT",
                "confidence": 1.0,
                "risk_level": "LOW" if primary_exit.get("emergency") else "MEDIUM",
                
                # NEW: Context-specific analysis fields
                "analysis_context": analysis_context,
                "entry_analysis": entry_analysis,
                "exit_cutloss_analysis": exit_cutloss_analysis,
                "exit_profit_analysis": exit_profit_analysis,
                
                # Backward compatibility
                "reasoning": active_analysis,
                
                "strategy_used": primary_exit.get("exit_type", "unknown"),
                "position_size_modifier": 1.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                
                # Exit signals use aggregate_id instead of trade_id
                "exit_signals": processed_exit_signals,
                "position_updates": [],
                "open_positions_count": len(processed_exit_signals),
                "signal_strength": 0,
                "key_factors": [signal.get("reason", "Unknown exit reason") for signal in processed_exit_signals],
                
                # Include primary aggregate_id if available
                "aggregate_id": primary_exit.get("aggregate_id", "")
            }
        }
    
    def _create_wait_decision(self, 
        reason: str, 
        analysis_context: str = "unknown",
        entry_analysis: Optional[str] = None,
        exit_cutloss_analysis: Optional[str] = None,
        exit_profit_analysis: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create wait/no action decision with context-specific analysis"""
        return {
            "current_recommendation": {
                "recommendation": "WAIT",
                "confidence": 0.0,
                "risk_level": "NONE",
                
                # NEW: Context-specific analysis fields
                "analysis_context": analysis_context,
                "entry_analysis": entry_analysis,
                "exit_cutloss_analysis": exit_cutloss_analysis,
                "exit_profit_analysis": exit_profit_analysis,
                
                # Backward compatibility
                "reasoning": reason,
                
                "strategy_used": "none",
                "position_size_modifier": 1.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "exit_signals": [],
                "position_updates": [],
                "open_positions_count": 0,
                "signal_strength": 0,
                "key_factors": [reason],
                "aggregate_id": ""  # Empty for wait decisions
            }
        }
    
    def list_available_strategies(self) -> Dict[str, List[str]]:
        """Return all available strategy identifiers"""
        return {
            "entry_strategies": list(self.ENTRY_STRATEGIES.keys()),
            "take_profit_strategies": list(self.TAKE_PROFIT_STRATEGIES.keys()),
            "cut_loss_strategies": list(self.CUT_LOSS_STRATEGIES.keys())
        }
