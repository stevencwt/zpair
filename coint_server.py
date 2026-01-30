#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
coint_server.py - Cointegration Testing API Server (Fixed)
Fixes: Handles Numpy types, NaN/Infinity values, CORS, and JSON truncation.
"""

import json
import argparse
import math
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Import existing cointegration modules
try:
    from pair_validator import PairValidator
except ImportError as e:
    print(f"⚠️  WARNING: Could not import core modules: {e}")
    print("    Ensure pair_validator.py is in the same directory.")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Configuration
CONFIG_FILE = 'config_cointegrate.json'
RESULTS_FILE = 'coint_results.json'
HTML_FILE = 'zcoint.html'

# --- CRITICAL FIX: Custom JSON Encoder for Numpy/NaN ---
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle Numpy Integers
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        # Handle Numpy Floats
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj) if math.isfinite(obj) else None
        # Handle Arrays
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        # Handle Standard Floats (NaN/Inf)
        elif isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        
        return super().default(obj)

# Register the custom encoder with Flask
app.json_provider_class = type('CustomJSONProvider', (app.json_provider_class,), {
    'default': staticmethod(CustomJSONEncoder().default)
})

class CointegrationServer:
    """Backend server for cointegration dashboard"""
    
    def __init__(self, config_path: str = CONFIG_FILE):
        self.config_path = Path(config_path)
        self.results_path = Path(RESULTS_FILE)
        self.html_path = Path(HTML_FILE)
        
        # Initialize validator
        self.validator = None
        self._init_validator()
    
    def _init_validator(self):
        """Initialize PairValidator from config"""
        try:
            config = self.load_config()
            self.validator = PairValidator(
                timeframe=config.get('timeframe', '5m'),
                lookback_bars=config.get('lookback_bars', 1000),
                significance_level=config.get('significance_level', 0.05)
            )
            print("✅ PairValidator Initialized")
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize validator: {e}")
    
    def load_config(self) -> Dict:
        """Load configuration from JSON file"""
        if not self.config_path.exists():
            default_config = {
                "timeframe": "5m",
                "lookback_bars": 1000,
                "significance_level": 0.05,
                "pairs_to_test": [
                    {"asset1": "SOL", "asset2": "MELANIA"},
                    {"asset1": "SOL", "asset2": "JUP"},
                    {"asset1": "ETH", "asset2": "BTC"}
                ]
            }
            self.save_config(default_config)
            return default_config
        
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def save_config(self, config: Dict):
        """Save configuration to JSON file"""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def run_cointegration_test(self) -> Dict:
        """Run cointegration test on all pairs"""
        config = self.load_config()
        pairs = config.get('pairs_to_test', [])
        
        if not pairs:
            return {'status': 'error', 'message': 'No pairs configured'}
        
        print(f"🔄 Running tests on {len(pairs)} pairs...")
        
        # Run validation
        results = self.validator.validate_multiple_pairs(pairs)
        
        # Calculate scores and rankings
        ranked_results = self._rank_results(results)
        
        # Count viable pairs
        viable_count = sum(1 for r in ranked_results if r['is_cointegrated'])
        
        # Prepare output
        output = {
            'timestamp': datetime.now().isoformat(),
            'timeframe': config.get('timeframe'),
            'lookback_bars': config.get('lookback_bars'),
            'pairs_tested': len(ranked_results),
            'viable_count': viable_count,
            'results': ranked_results,
            'best_pair': ranked_results[0]['pair'] if ranked_results else None
        }
        
        # Save results
        self.save_results(output)
        
        return output
    
    def _rank_results(self, results: List) -> List[Dict]:
        """Rank results by composite score"""
        scored_results = []
        
        for result in results:
            if not result: continue
            
            # Calculate composite score
            score = self._calculate_score(result)
            
            # Convert to dict manually to ensure keys exist
            result_dict = {
                'asset1': result.asset1,
                'asset2': result.asset2,
                'pair': f"{result.asset1}/{result.asset2}",
                'is_cointegrated': bool(result.is_cointegrated),
                'p_value': result.p_value,
                'adf_statistic': result.adf_statistic,
                'half_life_bars': result.half_life,
                # Convert half-life to minutes (assuming 5m timeframe if not specified)
                'half_life_minutes': (result.half_life * 5) if math.isfinite(result.half_life) else None,
                'hedge_ratio': result.hedge_ratio,
                'recommendation': result.recommendation,
                'samples_used': result.samples_used,
                'score': score
            }
            
            scored_results.append(result_dict)
        
        # Sort by score (descending)
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Add rank and position_value
        total = len(scored_results)
        for i, result in enumerate(scored_results, 1):
            result['rank'] = i
            result['position_value'] = self._calculate_position_value(i, total)
        
        return scored_results
    
    def _calculate_score(self, result) -> float:
        """Calculate composite score (0.0 to 1.0)"""
        if not result.is_cointegrated:
            return 0.0
        
        # P-value component (50% weight) - Lower is better
        p_val = result.p_value if math.isfinite(result.p_value) else 1.0
        p_score = max(0, 50 * (1 - p_val / 0.05))
        
        # Half-life component (30% weight) - Shorter is better
        hl = result.half_life if math.isfinite(result.half_life) else 9999
        if hl < 6: hl_score = 30
        elif hl < 24: hl_score = 20
        elif hl < 72: hl_score = 10
        else: hl_score = 5
        
        # Recommendation component (20% weight)
        rec_scores = {'EXCELLENT': 20, 'GOOD': 15, 'MODERATE': 10, 'WEAK': 5, 'SLOW': 3}
        rec_score = rec_scores.get(result.recommendation, 0)
        
        return round((p_score + hl_score + rec_score) / 100.0, 4)
    
    def _calculate_position_value(self, rank: int, total: int) -> float:
        """Calculate position value (1.0 = best)"""
        if total <= 1: return 1.0
        return round(1.0 - ((rank - 1) / (total - 1)), 2)
    
    def save_results(self, results: Dict):
        """Save results using Custom Encoder"""
        with open(self.results_path, 'w') as f:
            # Explicitly use cls=CustomJSONEncoder
            json.dump(results, f, indent=2, cls=CustomJSONEncoder)
    
    def load_results(self) -> Dict:
        """Load results safely"""
        if not self.results_path.exists():
            return None
        try:
            with open(self.results_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("⚠️  Warning: Result file was corrupted. Returning None.")
            return None
    
    def add_pair(self, asset1: str, asset2: str) -> Dict:
        """Add a pair to config"""
        config = self.load_config()
        pairs = config.get('pairs_to_test', [])
        for pair in pairs:
            if pair['asset1'] == asset1 and pair['asset2'] == asset2:
                return {'status': 'error', 'message': 'Pair already exists'}
        pairs.append({'asset1': asset1, 'asset2': asset2})
        config['pairs_to_test'] = pairs
        self.save_config(config)
        return {'status': 'success', 'message': f'Added {asset1}/{asset2}'}

    def remove_pair(self, asset1: str, asset2: str) -> Dict:
        """Remove a pair from config"""
        config = self.load_config()
        pairs = config.get('pairs_to_test', [])
        pairs = [p for p in pairs if not (p['asset1'] == asset1 and p['asset2'] == asset2)]
        config['pairs_to_test'] = pairs
        self.save_config(config)
        return {'status': 'success', 'message': f'Removed {asset1}/{asset2}'}

# Initialize server
server = CointegrationServer()

# API Routes
@app.route('/')
def serve_html():
    if server.html_path.exists():
        return send_from_directory('.', server.html_path.name)
    return "Dashboard HTML not found.", 404

@app.route('/api/results', methods=['GET'])
def get_results():
    results = server.load_results()
    if results is None:
        return jsonify({'error': 'No results available'}), 404
    return jsonify(results)

@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify(server.load_config())

@app.route('/api/run-test', methods=['POST'])
def run_test():
    try:
        results = server.run_cointegration_test()
        return jsonify({
            'status': 'success', 
            'message': f"Tested {results['pairs_tested']} pairs",
            'viable_count': results['viable_count']
        })
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/add-pair', methods=['POST'])
def add_pair_route():
    data = request.json
    return jsonify(server.add_pair(data.get('asset1'), data.get('asset2')))

@app.route('/api/remove-pair', methods=['POST'])
def remove_pair_route():
    data = request.json
    return jsonify(server.remove_pair(data.get('asset1'), data.get('asset2')))

def main():
    parser = argparse.ArgumentParser(description='Cointegration Server')
    parser.add_argument('--port', type=int, default=5001)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    print(f"\n🚀 Server running at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()