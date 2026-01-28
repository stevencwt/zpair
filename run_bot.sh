#!/bin/bash

# ==============================================================================
# V14s TRADING SYSTEM LAUNCHER (MAC) - FIXED TIMING
# ==============================================================================

cleanup() {
    echo ""
    echo "🛑 Shutting down all trading services..."
    kill $PID_HTTP 2>/dev/null
    kill $PID_SCHED 2>/dev/null
    kill $PID_CAF 2>/dev/null
    echo "✅ Shutdown complete."
    exit
}

trap cleanup SIGINT

echo "☕ Starting Caffeinate (Preventing Sleep)..."
caffeinate -i &
PID_CAF=$!

echo "🌐 Starting HTTP Server on Port 8000..."
python3 -m http.server 8000 > /dev/null 2>&1 &
PID_HTTP=$!
echo "   -> Dashboard: http://localhost:8000/zsimple.html"

# [FIX] Start Enforcer with a 15-second delay to allow VMain to boot first
echo "⏰ Scheduling Enforcer (Starts in 15s)..."
(sleep 15 && python3 schedule_enforcer.py) &
PID_SCHED=$!

echo "🚀 Starting V14s Main Engine..."
echo "==================================================="
python3 vmain_v14s.py --config config_SOL_MELANIA_5m1m.json

cleanup