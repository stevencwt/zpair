#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Market Barometer Module - Enhanced reusable market analysis system
Analyzes market conditions across major cryptocurrencies and provides trading recommendations
Updated to use HyperliquidAdapter from hyperliquid_utils_adapter.py

Changes from original:
1. Updated to output risk_on/risk_off/neutral state instead of 5-state system
2. Enhanced confidence calculation with multiple factors
3. Compatible with HyperliquidAdapter interface
4. Improved classification logic for trend-following strategies
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum


class MarketState(Enum):
    """Market state for trend-following strategies"""
    RISK_ON = "risk_on"      # Favor trend-following longs
    RISK_OFF = "risk_off"    # Favor trend-following shorts
    NEUTRAL = "neutral"      # Sideways/mixed market, reduce size


@dataclass
class MarketData:
    symbol: str
    current_price: float
    price_1h_ago: float
    price_4h_ago: float
    price_24h_ago: float
    weight: float = 1.0
    
    def pct_change_1h(self) -> float:
        if self.price_1h_ago == 0:
            return 0.0
        return ((self.current_price - self.price_1h_ago) / self.price_1h_ago) * 100
    
    def pct_change_4h(self) -> float:
        if self.price_4h_ago == 0:
            return 0.0
        return ((self.current_price - self.price_4h_ago) / self.price_4h_ago) * 100
    
    def pct_change_24h(self) -> float:
        if self.price_24h_ago == 0:
            return 0.0
        return ((self.current_price - self.price_24h_ago) / self.price_24h_ago) * 100


@dataclass
class MarketBarometer:
    """Market barometer with enhanced state system"""
    state: MarketState           # risk_on/risk_off/neutral  
    confidence: float           # 0.0 to 1.0 conviction level
    score: float               # Composite score for debugging (-100 to +100)
    bull_coins_count: int
    bear_coins_count: int
    neutral_coins_count: int
    dominant_timeframe: str
    trend_strength: float      # 0.0 to 1.0 trending vs choppy
    major_coin_alignment: float  # BTC/ETH alignment factor
    analysis_details: Dict


class MarketAnalyzer:
    """Enhanced market analyzer with risk_on/risk_off classification"""
    
    def __init__(self, hl_adapter, market_config: Dict = None):
        """
        Initialize market analyzer
        
        Args:
            hl_adapter: HyperliquidAdapter instance from hyperliquid_utils_adapter.py
            market_config: Configuration dictionary with settings
        """
        self.hl_adapter = hl_adapter
        
        # Use provided config or defaults
        if market_config is None:
            market_config = {}
        
        # Major cryptocurrencies with their market influence weights
        # BTC/ETH have highest weights as they drive macro trends
        self.major_coins = market_config.get("major_coins", {
            'BTC': 4.0,    # Highest weight - ultimate market leader
            'ETH': 3.0,    # Second most influential - institutional favorite  
            'SOL': 2.0,    # Major L1 alt with high momentum correlation
            'BNB': 1.5,    # Exchange token with significant influence
            'XRP': 1.2,    # Established coin with institutional adoption
            'ADA': 1.0,    # Major alt that follows broader trends
            'AVAX': 1.0,   # Layer 1 with DeFi correlation
            'MATIC': 0.8,  # Layer 2 scaling narrative
            'DOT': 0.8,    # Interoperability narrative
            'LINK': 0.8    # Oracle/infrastructure narrative
        })
        
        # Enhanced thresholds for risk_on/risk_off classification
        self.state_thresholds = market_config.get("state_thresholds", {
            'risk_on_threshold': 5.0,      # Strong bullish composite score
            'risk_off_threshold': -5.0,    # Strong bearish composite score
            # Between -5.0 and +5.0 = neutral range
        })
        
        # Confidence calculation weights
        self.confidence_weights = market_config.get("confidence_weights", {
            'consensus_weight': 0.3,        # Coin consensus alignment
            'score_magnitude_weight': 0.25, # Strength of composite score
            'timeframe_alignment_weight': 0.25, # Cross-timeframe consistency
            'major_coin_weight': 0.2        # BTC/ETH specific influence
        })
        
        # Cache for price history to avoid excessive API calls
        self.price_cache = {}
        self.cache_timestamp = 0
        self.cache_duration = 300  # 5 minutes
        
        # Verbose logging flag
        self.verbose = market_config.get("verbose", False)
    
    def get_historical_prices(self, symbol: str, hours_back: List[int]) -> Dict[int, float]:
        """Get historical prices for a symbol at specified hours back"""
        try:
            prices = {}
            
            for hours in hours_back:
                # Get candles for the period  
                bars_needed = hours + 1  # Extra buffer
                candles = self.hl_adapter.get_candles_snapshot(symbol, "1h", bars_needed)
                
                if candles and len(candles) > hours:
                    # Get price from hours_back ago
                    target_candle = candles[-(hours + 1)] if hours > 0 else candles[-1]
                    prices[hours] = float(target_candle.get('c', 0))
                else:
                    prices[hours] = 0.0
                    
            return prices
            
        except Exception as e:
            if self.verbose:
                print(f"[market] Error getting historical prices for {symbol}: {e}")
            return {h: 0.0 for h in hours_back}
    
    def collect_market_data(self) -> List[MarketData]:
        """Collect current and historical price data for major cryptocurrencies"""
        current_time = time.time()
        
        # Check cache validity
        if (current_time - self.cache_timestamp) < self.cache_duration and self.price_cache:
            return self.price_cache.get('market_data', [])
        
        market_data = []
        
        for symbol, weight in self.major_coins.items():
            try:
                # Get current price using adapter's last_price method
                current_price = self.hl_adapter.last_price(symbol)
                if current_price is None:
                    continue
                
                # Get historical prices
                historical_prices = self.get_historical_prices(symbol, [1, 4, 24])
                
                data = MarketData(
                    symbol=symbol,
                    current_price=current_price,
                    price_1h_ago=historical_prices.get(1, current_price),
                    price_4h_ago=historical_prices.get(4, current_price),
                    price_24h_ago=historical_prices.get(24, current_price),
                    weight=weight
                )
                
                market_data.append(data)
                
            except Exception as e:
                if self.verbose:
                    print(f"[market] Error collecting data for {symbol}: {e}")
                continue
        
        # Update cache
        self.price_cache = {'market_data': market_data}
        self.cache_timestamp = current_time
        
        return market_data
    
    def calculate_weighted_market_score(self, market_data: List[MarketData]) -> Tuple[float, Dict]:
        """Calculate weighted market sentiment score across multiple timeframes"""
        if not market_data:
            return 0.0, {}
        
        timeframes = ['1h', '4h', '24h']
        # Longer timeframes more important for trend identification
        timeframe_weights = {'1h': 0.3, '4h': 1.0, '24h': 1.7}  
        
        timeframe_scores = {}
        analysis_details = {}
        
        for timeframe in timeframes:
            weighted_changes = []
            total_weight = 0.0
            
            for data in market_data:
                if timeframe == '1h':
                    pct_change = data.pct_change_1h()
                elif timeframe == '4h':
                    pct_change = data.pct_change_4h()
                else:  # 24h
                    pct_change = data.pct_change_24h()
                
                weighted_change = pct_change * data.weight
                weighted_changes.append(weighted_change)
                total_weight += data.weight
            
            # Calculate weighted average for this timeframe
            if total_weight > 0:
                timeframe_score = sum(weighted_changes) / total_weight
            else:
                timeframe_score = 0.0
            
            timeframe_scores[timeframe] = timeframe_score
            
            # Store details for analysis
            analysis_details[f'{timeframe}_score'] = timeframe_score
            analysis_details[f'{timeframe}_active_coins'] = len([d for d in market_data if 
                (d.pct_change_1h() if timeframe == '1h' else 
                 d.pct_change_4h() if timeframe == '4h' else d.pct_change_24h()) != 0])
        
        # Calculate composite score using timeframe weights
        total_timeframe_weight = sum(timeframe_weights.values())
        composite_score = sum(
            timeframe_scores[tf] * timeframe_weights[tf] 
            for tf in timeframes
        ) / total_timeframe_weight
        
        # Determine dominant timeframe (highest absolute score)
        dominant_timeframe = max(timeframe_scores.keys(), 
                               key=lambda x: abs(timeframe_scores[x]))
        
        analysis_details['composite_score'] = composite_score
        analysis_details['dominant_timeframe'] = dominant_timeframe
        analysis_details['timeframe_scores'] = timeframe_scores
        
        return composite_score, analysis_details
    
    def calculate_trend_strength(self, market_data: List[MarketData]) -> float:
        """Calculate trend strength vs choppiness (0.0 = choppy, 1.0 = trending)"""
        if not market_data:
            return 0.0
            
        # Count coins with strong directional moves (> 3% in 24h)
        strong_moves = 0
        total_coins = len(market_data)
        
        for data in market_data:
            change_24h = abs(data.pct_change_24h())
            if change_24h > 3.0:  # Strong directional move
                strong_moves += 1
                
        # Trend strength based on percentage of coins making strong moves
        trend_strength = strong_moves / total_coins if total_coins > 0 else 0.0
        
        return min(trend_strength * 1.5, 1.0)  # Amplify but cap at 1.0
    
    def calculate_major_coin_alignment(self, market_data: List[MarketData]) -> float:
        """Calculate BTC/ETH alignment factor (0.0 = divergent, 1.0 = aligned)"""
        btc_data = None
        eth_data = None
        
        for data in market_data:
            if data.symbol == 'BTC':
                btc_data = data
            elif data.symbol == 'ETH':
                eth_data = data
                
        if not btc_data or not eth_data:
            return 0.5  # Neutral if we can't find both
            
        # Check alignment across timeframes
        alignments = []
        
        for timeframe in ['1h', '4h', '24h']:
            if timeframe == '1h':
                btc_change = btc_data.pct_change_1h()
                eth_change = eth_data.pct_change_1h()
            elif timeframe == '4h':
                btc_change = btc_data.pct_change_4h()
                eth_change = eth_data.pct_change_4h()
            else:  # 24h
                btc_change = btc_data.pct_change_24h()
                eth_change = eth_data.pct_change_24h()
                
            # Check if both are moving in same direction with meaningful magnitude
            if abs(btc_change) > 1.0 and abs(eth_change) > 1.0:
                # Both moving significantly, check alignment
                same_direction = (btc_change > 0) == (eth_change > 0)
                alignments.append(1.0 if same_direction else 0.0)
            else:
                # Not much movement, neutral alignment
                alignments.append(0.5)
                
        return sum(alignments) / len(alignments) if alignments else 0.5
    
    def calculate_enhanced_confidence(self, market_data: List[MarketData], composite_score: float, 
                                    timeframe_scores: Dict, trend_strength: float, 
                                    major_coin_alignment: float) -> float:
        """Calculate enhanced confidence using multiple factors"""
        
        factors = []
        weights = self.confidence_weights
        
        # Factor 1: Consensus among coins (existing approach)
        bull_24h = len([d for d in market_data if d.pct_change_24h() > 2.0])
        bear_24h = len([d for d in market_data if d.pct_change_24h() < -2.0])
        total_coins = len(market_data)
        
        if total_coins > 0:
            consensus_ratio = max(bull_24h, bear_24h) / total_coins
            consensus_strength = min(consensus_ratio * 1.5, 1.0)
        else:
            consensus_strength = 0.0
            
        factors.append(consensus_strength * weights['consensus_weight'])
        
        # Factor 2: Score magnitude (stronger moves = higher confidence)
        score_strength = min(abs(composite_score) / 10.0, 1.0)
        factors.append(score_strength * weights['score_magnitude_weight'])
        
        # Factor 3: Timeframe alignment (all timeframes pointing same direction)  
        score_signs = [1 if score > 1.0 else -1 if score < -1.0 else 0 
                      for score in timeframe_scores.values()]
        
        if len(set(score_signs)) == 1 and 0 not in score_signs:
            timeframe_alignment = 1.0  # Perfect alignment
        elif 0 in score_signs:
            timeframe_alignment = 0.3  # Some neutral timeframes
        else:
            timeframe_alignment = 0.0  # Conflicting signals
            
        factors.append(timeframe_alignment * weights['timeframe_alignment_weight'])
        
        # Factor 4: Major coins (BTC/ETH) alignment strength
        factors.append(major_coin_alignment * weights['major_coin_weight'])
        
        # Weighted confidence calculation
        confidence = sum(factors)
        
        # Apply trend strength bonus - trending markets are more predictable
        confidence = confidence + (trend_strength * 0.1)  # Small bonus for trending
        
        return min(confidence, 1.0)
    
    def classify_market_state(self, score: float, confidence: float, market_data: List[MarketData], 
                            trend_strength: float, major_coin_alignment: float) -> Tuple[MarketState, Dict]:
        """Classify market state based on enhanced analysis"""
        
        # Count bullish/bearish coins across timeframes for analysis
        bull_1h = len([d for d in market_data if d.pct_change_1h() > 1.0])
        bear_1h = len([d for d in market_data if d.pct_change_1h() < -1.0])
        neutral_1h = len(market_data) - bull_1h - bear_1h
        
        bull_24h = len([d for d in market_data if d.pct_change_24h() > 2.0])
        bear_24h = len([d for d in market_data if d.pct_change_24h() < -2.0])
        neutral_24h = len(market_data) - bull_24h - bear_24h
        
        # Enhanced state classification with confidence weighting
        min_confidence_for_directional = 0.4  # Minimum confidence for risk_on/risk_off
        
        if score >= self.state_thresholds['risk_on_threshold']:
            if confidence >= min_confidence_for_directional:
                state = MarketState.RISK_ON
            else:
                state = MarketState.NEUTRAL  # Low confidence, stay neutral
        elif score <= self.state_thresholds['risk_off_threshold']:
            if confidence >= min_confidence_for_directional:
                state = MarketState.RISK_OFF
            else:
                state = MarketState.NEUTRAL  # Low confidence, stay neutral
        else:
            state = MarketState.NEUTRAL
        
        classification_details = {
            'bull_1h_count': bull_1h,
            'bear_1h_count': bear_1h,
            'neutral_1h_count': neutral_1h,
            'bull_24h_count': bull_24h,
            'bear_24h_count': bear_24h,
            'neutral_24h_count': neutral_24h,
            'total_coins_analyzed': len(market_data),
            'trend_strength': trend_strength,
            'major_coin_alignment': major_coin_alignment,
            'confidence_threshold_met': confidence >= min_confidence_for_directional
        }
        
        return state, classification_details
    
    def analyze_market(self) -> MarketBarometer:
        """Perform complete market analysis and return barometer reading"""
        try:
            # Collect market data
            market_data = self.collect_market_data()
            
            if not market_data:
                return MarketBarometer(
                    state=MarketState.NEUTRAL,
                    confidence=0.0,
                    score=0.0,
                    bull_coins_count=0,
                    bear_coins_count=0,
                    neutral_coins_count=0,
                    dominant_timeframe="unknown",
                    trend_strength=0.0,
                    major_coin_alignment=0.0,
                    analysis_details={"error": "No market data available"}
                )
            
            # Calculate composite market score
            composite_score, score_details = self.calculate_weighted_market_score(market_data)
            
            # Calculate additional metrics
            trend_strength = self.calculate_trend_strength(market_data)
            major_coin_alignment = self.calculate_major_coin_alignment(market_data)
            
            # Calculate enhanced confidence
            confidence = self.calculate_enhanced_confidence(
                market_data, composite_score, score_details['timeframe_scores'], 
                trend_strength, major_coin_alignment
            )
            
            # Classify market state
            state, classification_details = self.classify_market_state(
                composite_score, confidence, market_data, trend_strength, major_coin_alignment
            )
            
            # Combine all analysis details
            all_details = {
                **score_details,
                **classification_details,
                'coins_analyzed': [d.symbol for d in market_data],
                'major_movers_up': [d.symbol for d in market_data if d.pct_change_24h() > 5.0],
                'major_movers_down': [d.symbol for d in market_data if d.pct_change_24h() < -5.0],
                'btc_24h_change': next((d.pct_change_24h() for d in market_data if d.symbol == 'BTC'), 0.0),
                'eth_24h_change': next((d.pct_change_24h() for d in market_data if d.symbol == 'ETH'), 0.0),
            }
            
            return MarketBarometer(
                state=state,
                confidence=round(confidence, 3),
                score=round(composite_score, 2),
                bull_coins_count=classification_details['bull_24h_count'],
                bear_coins_count=classification_details['bear_24h_count'],
                neutral_coins_count=classification_details['neutral_24h_count'],
                dominant_timeframe=score_details['dominant_timeframe'],
                trend_strength=round(trend_strength, 3),
                major_coin_alignment=round(major_coin_alignment, 3),
                analysis_details=all_details
            )
            
        except Exception as e:
            if self.verbose:
                print(f"[market] Error in market analysis: {e}")
            return MarketBarometer(
                state=MarketState.NEUTRAL,
                confidence=0.0,
                score=0.0,
                bull_coins_count=0,
                bear_coins_count=0,
                neutral_coins_count=0,
                dominant_timeframe="error",
                trend_strength=0.0,
                major_coin_alignment=0.0,
                analysis_details={"error": str(e)}
            )


def format_market_status(barometer: MarketBarometer) -> str:
    """Format market barometer for display"""
    state_emojis = {
        MarketState.RISK_ON: "🚀",
        MarketState.RISK_OFF: "🔻", 
        MarketState.NEUTRAL: "➡️"
    }
    
    emoji = state_emojis.get(barometer.state, "❓")
    state_name = barometer.state.value.replace('_', ' ').upper()
    
    confidence_text = f"{barometer.confidence * 100:.0f}%"
    trend_text = f"Trend: {barometer.trend_strength * 100:.0f}%"
    
    return f"{emoji} {state_name} | Score: {barometer.score:+.1f} | Confidence: {confidence_text} | {trend_text}"


def get_trading_recommendation(barometer: MarketBarometer, min_confidence: float = 0.6) -> Dict[str, str]:
    """
    Generate trading recommendations based on market barometer
    
    Args:
        barometer: MarketBarometer object
        min_confidence: Minimum confidence level for recommendations
        
    Returns:
        Dictionary with trading recommendations
    """
    
    recommendations = {
        'primary_recommendation': 'NEUTRAL',
        'size_suggestion': 'REDUCE',
        'strategy_type': 'RANGE/MEAN_REVERSION',
        'reasoning': 'Low confidence or neutral market conditions',
        'confidence_check': 'FAIL'
    }
    
    if barometer.confidence < min_confidence:
        recommendations['reasoning'] = f'Low confidence ({barometer.confidence:.2f} < {min_confidence})'
        return recommendations
    
    recommendations['confidence_check'] = 'PASS'
    
    if barometer.state == MarketState.RISK_ON:
        recommendations.update({
            'primary_recommendation': 'RISK_ON',
            'size_suggestion': 'INCREASE' if barometer.confidence > 0.8 else 'NORMAL',
            'strategy_type': 'TREND_FOLLOWING_LONG',
            'reasoning': f'Risk-on market with {barometer.confidence:.2f} confidence'
        })
        
    elif barometer.state == MarketState.RISK_OFF:
        recommendations.update({
            'primary_recommendation': 'RISK_OFF',
            'size_suggestion': 'INCREASE' if barometer.confidence > 0.8 else 'NORMAL',
            'strategy_type': 'TREND_FOLLOWING_SHORT',
            'reasoning': f'Risk-off market with {barometer.confidence:.2f} confidence'
        })
        
    else:  # NEUTRAL
        if barometer.trend_strength < 0.3:
            recommendations.update({
                'primary_recommendation': 'NEUTRAL',
                'size_suggestion': 'REDUCE',
                'strategy_type': 'RANGE/MEAN_REVERSION',
                'reasoning': f'Choppy market (trend strength: {barometer.trend_strength:.2f})'
            })
        else:
            recommendations.update({
                'primary_recommendation': 'WAIT',
                'size_suggestion': 'REDUCE',
                'strategy_type': 'WAIT_FOR_BREAKOUT',
                'reasoning': f'Trending but neutral score (trend: {barometer.trend_strength:.2f})'
            })
    
    return recommendations


# Example usage function
def example_usage():
    """Example of how to use the updated Market Barometer with HyperliquidAdapter"""
    
    # Import the adapter (you need to have hyperliquid_utils_adapter.py in your path)
    # from hyperliquid_utils_adapter import HyperliquidAdapter
    
    # Initialize the adapter with your credentials
    # hl_adapter = HyperliquidAdapter(
    #     private_key_hex="your_private_key_here",
    #     wallet_address="your_wallet_address_here"
    # )
    
    # Optional custom configuration
    # custom_config = {
    #     "major_coins": {
    #         'BTC': 4.0,
    #         'ETH': 3.0,
    #         'SOL': 2.0,
    #         'BNB': 1.5
    #     },
    #     "state_thresholds": {
    #         'risk_on_threshold': 7.0,    # Higher threshold = more conservative
    #         'risk_off_threshold': -7.0
    #     },
    #     "verbose": True
    # }
    
    # Initialize the market analyzer
    # market_analyzer = MarketAnalyzer(hl_adapter, custom_config)
    
    # Perform market analysis
    # barometer = market_analyzer.analyze_market()
    
    # Display results
    # print("Market Status:", format_market_status(barometer))
    # print("State:", barometer.state.value)
    # print("Confidence:", f"{barometer.confidence:.3f}")
    # print("Trend Strength:", f"{barometer.trend_strength:.3f}")
    
    # Get trading recommendations
    # recommendations = get_trading_recommendation(barometer)
    # print("Recommendation:", recommendations['primary_recommendation'])
    # print("Strategy:", recommendations['strategy_type'])
    # print("Reasoning:", recommendations['reasoning'])
    
    pass


if __name__ == "__main__":
    print("Market Barometer Module - Updated for HyperliquidAdapter")
    print("Import this module and use MarketAnalyzer class with your HyperliquidAdapter instance")
    example_usage()