#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
symbol_utils.py - Symbol Classification and Parsing Utilities
=============================================================
Utilities for parsing and classifying trading symbols and asset pairs.
"""

from typing import Dict, Any, Tuple, Optional
import re


class SymbolClassifier:
    """Utility class for classifying and parsing trading symbols"""
    
    # Standard stable coin and fiat quote assets
    STABLE_QUOTES = {
        "USDT", "USDC", "USD", "BUSD", "USDX", "USDC.E", 
        "DAI", "TUSD", "SUSD", "GUSD", "PAX", "PAXG"
    }
    
    # Major quote assets (including stables)
    MAJOR_QUOTES = STABLE_QUOTES | {
        "BTC", "ETH", "BNB", "SOL", "MATIC", "AVAX", "DOT", "ADA"
    }
    
    @classmethod
    def classify_symbol(cls, symbol: str) -> Dict[str, Any]:
        """
        Classify symbol as single asset or pair with comprehensive metadata
        
        Args:
            symbol: Symbol string (e.g., "SOL/MELANIA", "BTC", "USDT")
            
        Returns:
            Dict containing classification metadata
        """
        if not symbol or not isinstance(symbol, str):
            return cls._create_invalid_symbol_result(symbol)
        
        # Clean the symbol
        symbol = symbol.strip().upper()
        
        # Parse base and quote
        base, quote = cls._parse_symbol_parts(symbol)
        
        # Determine symbol type
        symbol_type = cls._determine_symbol_type(base, quote)
        
        # Get additional metadata
        metadata = cls._get_symbol_metadata(base, quote, symbol_type)
        
        return {
            "type": symbol_type,
            "base": base,
            "quote": quote,
            "display": symbol,
            "original": symbol,
            "is_pair": symbol_type == "pair",
            "is_single": symbol_type == "single",
            "is_valid": True,
            "metadata": metadata
        }
    
    @classmethod
    def _parse_symbol_parts(cls, symbol: str) -> Tuple[str, str]:
        """Parse symbol into base and quote parts"""
        if "/" in symbol:
            parts = symbol.split("/", 1)
            return parts[0].strip(), parts[1].strip()
        elif "-" in symbol:
            # Alternative delimiter
            parts = symbol.split("-", 1)
            return parts[0].strip(), parts[1].strip()
        else:
            # Single asset
            return symbol, ""
    
    @classmethod
    def _determine_symbol_type(cls, base: str, quote: str) -> str:
        """Determine if symbol represents a pair or single asset"""
        if not quote:
            return "single"
        elif quote in cls.STABLE_QUOTES:
            return "single"  # Base/Stable pairs treated as single asset pricing
        else:
            return "pair"  # True trading pair
    
    @classmethod
    def _get_symbol_metadata(cls, base: str, quote: str, symbol_type: str) -> Dict[str, Any]:
        """Get additional metadata about the symbol"""
        metadata = {
            "base_type": cls._classify_asset_type(base),
            "quote_type": cls._classify_asset_type(quote) if quote else None,
            "is_major_pair": quote in cls.MAJOR_QUOTES if quote else False,
            "is_stable_quoted": quote in cls.STABLE_QUOTES if quote else False,
            "is_crypto_pair": symbol_type == "pair" and quote not in cls.STABLE_QUOTES,
            "trading_category": cls._get_trading_category(base, quote, symbol_type)
        }
        
        return metadata
    
    @classmethod
    def _classify_asset_type(cls, asset: str) -> str:
        """Classify individual asset type"""
        if not asset:
            return "unknown"
        
        if asset in cls.STABLE_QUOTES:
            return "stablecoin"
        elif asset in {"BTC", "ETH", "BNB", "SOL", "MATIC", "AVAX", "DOT", "ADA"}:
            return "major_crypto"
        elif asset in {"USD", "EUR", "GBP", "JPY", "CAD", "AUD"}:
            return "fiat"
        else:
            return "altcoin"
    
    @classmethod
    def _get_trading_category(cls, base: str, quote: str, symbol_type: str) -> str:
        """Get trading category for the symbol"""
        if symbol_type == "single":
            if not quote:
                return "single_asset"
            elif quote in cls.STABLE_QUOTES:
                return "stable_quoted"
            else:
                return "major_quoted"
        else:  # pair
            if base in cls.MAJOR_QUOTES and quote in cls.MAJOR_QUOTES:
                return "major_pair"
            elif quote in cls.MAJOR_QUOTES:
                return "alt_major_pair"
            else:
                return "alt_alt_pair"
    
    @classmethod
    def _create_invalid_symbol_result(cls, symbol: Any) -> Dict[str, Any]:
        """Create result for invalid symbol"""
        return {
            "type": "invalid",
            "base": "",
            "quote": "",
            "display": str(symbol) if symbol else "",
            "original": symbol,
            "is_pair": False,
            "is_single": False,
            "is_valid": False,
            "metadata": {
                "error": "Invalid symbol format",
                "base_type": "unknown",
                "quote_type": "unknown",
                "is_major_pair": False,
                "is_stable_quoted": False,
                "is_crypto_pair": False,
                "trading_category": "invalid"
            }
        }
    
    @classmethod
    def is_pair_symbol(cls, symbol: str) -> bool:
        """Quick check if symbol represents a trading pair"""
        classification = cls.classify_symbol(symbol)
        return classification["is_pair"]
    
    @classmethod
    def is_valid_symbol(cls, symbol: str) -> bool:
        """Check if symbol is valid"""
        classification = cls.classify_symbol(symbol)
        return classification["is_valid"]
    
    @classmethod
    def get_base_asset(cls, symbol: str) -> str:
        """Extract base asset from symbol"""
        classification = cls.classify_symbol(symbol)
        return classification["base"]
    
    @classmethod
    def get_quote_asset(cls, symbol: str) -> str:
        """Extract quote asset from symbol"""
        classification = cls.classify_symbol(symbol)
        return classification["quote"]
    
    @classmethod
    def format_symbol(cls, base: str, quote: str, delimiter: str = "/") -> str:
        """Format symbol from base and quote assets"""
        if not base:
            return ""
        if not quote:
            return base.upper()
        return f"{base.upper()}{delimiter}{quote.upper()}"
    
    @classmethod
    def normalize_symbol(cls, symbol: str) -> str:
        """Normalize symbol to standard format"""
        classification = cls.classify_symbol(symbol)
        if not classification["is_valid"]:
            return symbol
        
        base = classification["base"]
        quote = classification["quote"]
        
        if quote:
            return f"{base}/{quote}"
        else:
            return base


class AssetPairValidator:
    """Validator for asset pairs in trading contexts"""
    
    @staticmethod
    def validate_pair_for_regression(base: str, quote: str) -> Dict[str, Any]:
        """
        Validate if asset pair is suitable for regression analysis
        
        Args:
            base: Base asset symbol
            quote: Quote asset symbol
            
        Returns:
            Validation result with recommendations
        """
        issues = []
        warnings = []
        
        # Check for same asset
        if base.upper() == quote.upper():
            issues.append("Base and quote assets are identical")
        
        # Check for stable pairs (usually low volatility)
        classifier = SymbolClassifier()
        base_type = classifier._classify_asset_type(base)
        quote_type = classifier._classify_asset_type(quote)
        
        if base_type == "stablecoin" and quote_type == "stablecoin":
            warnings.append("Both assets are stablecoins - may have low volatility")
        
        # Check for correlation likelihood
        correlation_warning = AssetPairValidator._assess_correlation_likelihood(base, quote)
        if correlation_warning:
            warnings.append(correlation_warning)
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "suitability_score": AssetPairValidator._calculate_suitability_score(base, quote, issues, warnings),
            "recommendations": AssetPairValidator._get_recommendations(base, quote, issues, warnings)
        }
    
    @staticmethod
    def _assess_correlation_likelihood(base: str, quote: str) -> Optional[str]:
        """Assess likelihood of correlation between assets"""
        # Same ecosystem pairs often correlate
        ecosystems = {
            "ethereum": ["ETH", "USDC", "DAI", "UNI", "AAVE", "COMP"],
            "solana": ["SOL", "USDC", "RAY", "SRM"],
            "binance": ["BNB", "BUSD", "CAKE"],
        }
        
        for ecosystem, assets in ecosystems.items():
            if base in assets and quote in assets:
                return f"Both assets from {ecosystem} ecosystem - may be highly correlated"
        
        return None
    
    @staticmethod
    def _calculate_suitability_score(base: str, quote: str, issues: list, warnings: list) -> float:
        """Calculate suitability score (0-1) for regression trading"""
        score = 1.0
        
        # Deduct for issues
        score -= len(issues) * 0.3
        
        # Deduct for warnings
        score -= len(warnings) * 0.1
        
        # Bonus for good combinations
        classifier = SymbolClassifier()
        base_type = classifier._classify_asset_type(base)
        quote_type = classifier._classify_asset_type(quote)
        
        if base_type == "altcoin" and quote_type == "major_crypto":
            score += 0.1  # Good for relative strength analysis
        
        return max(0.0, min(1.0, score))
    
    @staticmethod
    def _get_recommendations(base: str, quote: str, issues: list, warnings: list) -> list:
        """Get recommendations for improving pair selection"""
        recommendations = []
        
        if issues:
            recommendations.append("Fix critical issues before proceeding")
        
        if len(warnings) > 1:
            recommendations.append("Consider alternative asset pairs with different characteristics")
        
        # Specific recommendations based on asset types
        classifier = SymbolClassifier()
        base_type = classifier._classify_asset_type(base)
        quote_type = classifier._classify_asset_type(quote)
        
        if base_type == "stablecoin":
            recommendations.append("Consider using a volatile asset as base for better signals")
        
        if quote_type == "altcoin":
            recommendations.append("Consider using a major crypto or stable as quote for more reliable reference")
        
        return recommendations


# Legacy compatibility function
def classify_symbol(symbol: str) -> Dict[str, Any]:
    """Legacy function for backward compatibility"""
    return SymbolClassifier.classify_symbol(symbol)
